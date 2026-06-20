"""Select visually useful xBD pairs for the Jalon 5 Aftermath demo.

The script ranks examples from existing split CSV files using ground-truth masks,
then optionally runs the current app inference pipeline to include model outputs.
It only writes under demo_assets/jalon5_demo_pairs/ by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "src").is_dir() and (candidate / "scripts").is_dir():
            return candidate
    return start


PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)
PROJECT_SRC = PROJECT_ROOT / "src"
PROJECT_APP = PROJECT_ROOT / "app"
for path in (PROJECT_SRC, PROJECT_APP):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from crisismap.data.xbd_dataset import XBDDatasetError, XBDPairDataset  # noqa: E402


CLASS_COLORS = {
    0: np.array([0, 0, 0], dtype=np.uint8),
    1: np.array([0, 170, 80], dtype=np.uint8),
    2: np.array([220, 40, 40], dtype=np.uint8),
}


@dataclass
class SplitSpec:
    label: str
    path: Path


@dataclass
class Candidate:
    pair_id: str
    split: str
    split_csv: Path
    disaster: str
    index: int
    background_pixels: int
    no_damage_pixels: int
    damaged_pixels: int
    building_pixels: int
    total_pixels: int
    building_ratio: float
    damaged_ratio_total: float
    damaged_ratio_building: float
    no_damage_ratio_building: float
    pre_post_diff_building: float
    passes_filter: bool
    visual_score: float
    final_score: float = 0.0
    model_score: float | None = None
    model_metrics: dict[str, float] | None = None
    damage_ratio_pred: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank and export Jalon 5 demo pairs for the Aftermath Streamlit app."
    )
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "data" / "raw" / "xbd" / "train")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--target-mode", default="3-class", choices=["3-class"])
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "demo_assets" / "jalon5_demo_pairs")
    parser.add_argument(
        "--split-csv",
        action="append",
        default=[],
        help=(
            "Optional split CSV. Can be provided multiple times. "
            "Use label=path or just path."
        ),
    )
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--max-samples-per-split", type=int, default=0)
    parser.add_argument("--min-damaged-pixels", type=int, default=800)
    parser.add_argument("--min-no-damage-pixels", type=int, default=1200)
    parser.add_argument("--min-building-ratio", type=float, default=0.01)
    parser.add_argument("--max-building-ratio", type=float, default=0.20)
    parser.add_argument("--min-damaged-ratio-building", type=float, default=0.10)
    parser.add_argument("--max-damaged-ratio-building", type=float, default=0.75)
    parser.add_argument("--run-model", action="store_true")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--building-threshold", type=float, default=0.60)
    parser.add_argument("--building-tta", choices=["none", "d4"], default="d4")
    parser.add_argument("--damage-tta", choices=["none", "d4"], default="d4")
    parser.add_argument("--contact-cell-width", type=int, default=260)
    return parser.parse_args()


def discover_splits(args: argparse.Namespace) -> list[SplitSpec]:
    if args.split_csv:
        specs: list[SplitSpec] = []
        for item in args.split_csv:
            if "=" in item:
                label, raw_path = item.split("=", 1)
                path = Path(raw_path)
            else:
                path = Path(item)
                label = path.parent.name
            specs.append(SplitSpec(label=label.strip() or path.parent.name, path=path))
        return specs

    split_dirs = [
        PROJECT_ROOT / "data" / "processed" / "splits",
        PROJECT_ROOT / "data" / "processed" / "splits_full",
        PROJECT_ROOT / "data" / "processed" / "splits_noleak_match_hist_all",
        PROJECT_ROOT / "data" / "processed" / "splits_noleak_full_train",
    ]
    specs = []
    seen: set[Path] = set()
    for split_name in ("test", "val", "train"):
        for split_dir in split_dirs:
            path = split_dir / f"{split_name}_pairs.csv"
            resolved = path.resolve()
            if path.exists() and resolved not in seen:
                specs.append(SplitSpec(label=split_name, path=path))
                seen.add(resolved)
                break
    return specs


def disaster_from_pair_id(pair_id: str) -> str:
    return pair_id.rsplit("_", 1)[0] if "_" in pair_id else pair_id.split("-", 1)[0]


def tensor_to_rgb(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    return (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)


def colorize_damage(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.int64)
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        out[mask == class_id] = color
    return out


def colorize_building(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    out[mask.astype(bool)] = np.array([0, 180, 220], dtype=np.uint8)
    return out


def overlay_damage(post: np.ndarray, mask: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    colors = colorize_damage(mask).astype(np.float32)
    opacity = np.zeros(mask.shape, dtype=np.float32)
    opacity[mask == 1] = 0.35
    opacity[mask == 2] = alpha
    opacity = opacity[:, :, None]
    return np.clip(post.astype(np.float32) * (1.0 - opacity) + colors * opacity, 0, 255).astype(np.uint8)


def safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


def triangular_score(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    center = (low + high) / 2.0
    radius = (high - low) / 2.0
    return max(0.0, 1.0 - abs(value - center) / radius)


def gaussian_score(value: float, center: float, sigma: float) -> float:
    if sigma <= 0:
        return 0.0
    return math.exp(-0.5 * ((value - center) / sigma) ** 2)


def sample_candidate(
    sample: dict[str, Any],
    split_spec: SplitSpec,
    index: int,
    args: argparse.Namespace,
) -> Candidate:
    target = sample["target"].detach().cpu().numpy().astype(np.int16)
    image = sample["image"]
    pair_id = str(sample["pair_id"])

    background_pixels = int((target == 0).sum())
    no_damage_pixels = int((target == 1).sum())
    damaged_pixels = int((target == 2).sum())
    building_pixels = no_damage_pixels + damaged_pixels
    total_pixels = int(target.size)

    building_ratio = building_pixels / total_pixels if total_pixels else 0.0
    damaged_ratio_total = damaged_pixels / total_pixels if total_pixels else 0.0
    damaged_ratio_building = damaged_pixels / building_pixels if building_pixels else 0.0
    no_damage_ratio_building = no_damage_pixels / building_pixels if building_pixels else 0.0

    pre = image[:3].detach().cpu().numpy()
    post = image[3:6].detach().cpu().numpy()
    building_mask = target > 0
    if building_mask.any():
        diff = np.abs(pre - post).mean(axis=0)
        pre_post_diff_building = float(diff[building_mask].mean())
    else:
        pre_post_diff_building = 0.0

    passes_filter = (
        damaged_pixels >= args.min_damaged_pixels
        and no_damage_pixels >= args.min_no_damage_pixels
        and args.min_building_ratio <= building_ratio <= args.max_building_ratio
        and args.min_damaged_ratio_building <= damaged_ratio_building <= args.max_damaged_ratio_building
    )

    balance_score = 1.0 - abs(damaged_ratio_building - 0.35) / 0.35
    balance_score = max(0.0, min(1.0, balance_score))
    building_score = triangular_score(building_ratio, args.min_building_ratio, args.max_building_ratio)
    damage_amount_score = gaussian_score(damaged_ratio_total, center=0.018, sigma=0.018)
    diff_score = max(0.0, min(1.0, pre_post_diff_building / 0.18))
    intact_score = max(0.0, min(1.0, no_damage_ratio_building / 0.35))
    visual_score = (
        0.34 * balance_score
        + 0.24 * building_score
        + 0.18 * damage_amount_score
        + 0.14 * diff_score
        + 0.10 * intact_score
    )
    if not passes_filter:
        visual_score *= 0.72

    return Candidate(
        pair_id=pair_id,
        split=split_spec.label,
        split_csv=split_spec.path,
        disaster=disaster_from_pair_id(pair_id),
        index=index,
        background_pixels=background_pixels,
        no_damage_pixels=no_damage_pixels,
        damaged_pixels=damaged_pixels,
        building_pixels=building_pixels,
        total_pixels=total_pixels,
        building_ratio=building_ratio,
        damaged_ratio_total=damaged_ratio_total,
        damaged_ratio_building=damaged_ratio_building,
        no_damage_ratio_building=no_damage_ratio_building,
        pre_post_diff_building=pre_post_diff_building,
        passes_filter=passes_filter,
        visual_score=visual_score,
    )


def scan_candidates(args: argparse.Namespace, split_specs: list[SplitSpec]) -> tuple[list[Candidate], dict[str, Any]]:
    candidates: list[Candidate] = []
    seen_pairs: set[str] = set()
    errors: list[str] = []

    for split_spec in split_specs:
        try:
            dataset = XBDPairDataset(
                root=args.root,
                split_csv=split_spec.path,
                image_size=args.image_size,
                target_mode=args.target_mode,
                augment_mode="none",
            )
        except XBDDatasetError as exc:
            errors.append(f"{split_spec.path}: {exc}")
            continue

        limit = len(dataset)
        if args.max_samples_per_split > 0:
            limit = min(limit, args.max_samples_per_split)
        for index in range(limit):
            try:
                sample = dataset[index]
            except Exception as exc:  # keep scanning if one pair is unreadable
                errors.append(f"{split_spec.path}#{index}: {exc}")
                continue
            pair_id = str(sample["pair_id"])
            if pair_id in seen_pairs:
                continue
            seen_pairs.add(pair_id)
            candidates.append(sample_candidate(sample, split_spec, index, args))

    metadata = {
        "scanned_at": datetime.now().isoformat(),
        "split_csvs": [str(spec.path) for spec in split_specs],
        "errors": errors,
    }
    return candidates, metadata


def rank_candidates(candidates: list[Candidate], top_k: int) -> list[Candidate]:
    ordered = sorted(candidates, key=lambda item: item.visual_score, reverse=True)
    selected: list[Candidate] = []
    disaster_counts: dict[str, int] = {}

    pool = [item for item in ordered if item.passes_filter]
    if len(pool) < top_k:
        extras = [item for item in ordered if item not in pool]
        pool.extend(extras)

    while pool and len(selected) < top_k:
        best_index = 0
        best_score = -1.0
        for index, candidate in enumerate(pool):
            penalty = 0.88 ** disaster_counts.get(candidate.disaster, 0)
            score = candidate.visual_score * penalty
            if candidate.passes_filter:
                score += 0.04
            if score > best_score:
                best_score = score
                best_index = index
        chosen = pool.pop(best_index)
        chosen.final_score = best_score
        selected.append(chosen)
        disaster_counts[chosen.disaster] = disaster_counts.get(chosen.disaster, 0) + 1

    return selected


def load_app_pipeline(args: argparse.Namespace):
    import streamlit_app as app  # imported lazily to avoid Streamlit dependency for mask-only scans

    device_name = resolve_device_name(args.device)
    device = torch.device(device_name)

    model_cfg = app.DAMAGE_MODELS["damage_champion_v2"]
    cfg = {
        "device": device_name,
        "model_cfg": model_cfg,
        "use_building": bool(app.BUILDING_MODULE_AVAILABLE and app.BUILDING_CHECKPOINT.exists()),
    }
    damage_model, building_model = app.prepare_models(cfg)
    return app, damage_model, building_model, device


def run_model_for_sample(
    app_module,
    damage_model,
    building_model,
    sample: dict[str, Any],
    target: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    use_building = bool(building_model is not None)
    inference = app_module.run_inference(
        damage_model,
        sample["image"],
        resolve_device_name(args.device),
        args.damage_tta,
        use_building,
        building_model,
        args.building_threshold,
        args.building_tta,
    )
    metrics = app_module.compute_metrics(inference["final_pred"], target)
    stats = app_module.prediction_stats(inference["final_pred"])
    return {"inference": inference, "metrics": metrics, "stats": stats}


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image.astype(np.uint8)).save(path)


def export_candidate_assets(
    candidate: Candidate,
    rank: int,
    args: argparse.Namespace,
    app_state: tuple[Any, Any, Any, torch.device] | None,
) -> dict[str, Any] | None:
    dataset = XBDPairDataset(
        root=args.root,
        split_csv=candidate.split_csv,
        image_size=args.image_size,
        target_mode=args.target_mode,
        augment_mode="none",
    )
    sample = dataset[candidate.index]
    pre = tensor_to_rgb(sample["image"][:3])
    post = tensor_to_rgb(sample["image"][3:6])
    target = sample["target"].detach().cpu().numpy().astype(np.int16)

    pair_dir = args.output_dir / f"{rank:02d}_{safe_filename(candidate.pair_id)}"
    write_png(pair_dir / "pre.png", pre)
    write_png(pair_dir / "post.png", post)
    write_png(pair_dir / "target.png", colorize_damage(target))
    write_png(pair_dir / "overlay_target.png", overlay_damage(post, target))

    model_result = None
    if app_state is not None:
        app_module, damage_model, building_model, _device = app_state
        model_result = run_model_for_sample(app_module, damage_model, building_model, sample, target, args)
        inference = model_result["inference"]
        write_png(pair_dir / "raw_damage.png", colorize_damage(inference["raw_pred"]))
        if inference["building_mask"] is not None:
            write_png(pair_dir / "building_mask.png", colorize_building(inference["building_mask"]))
        write_png(pair_dir / "final_damage.png", colorize_damage(inference["final_pred"]))
        write_png(pair_dir / "overlay_model.png", overlay_damage(post, inference["final_pred"]))
        metrics_payload = {
            "pair_id": candidate.pair_id,
            "split": candidate.split,
            "damage_tta": args.damage_tta,
            "building_tta": args.building_tta,
            "building_threshold": args.building_threshold,
            "metrics": model_result["metrics"],
            "prediction_stats": model_result["stats"],
        }
        with (pair_dir / "metrics.json").open("w", encoding="utf-8") as file:
            json.dump(metrics_payload, file, indent=2, ensure_ascii=False)
        candidate.model_metrics = {
            key: float(value)
            for key, value in model_result["metrics"].items()
            if isinstance(value, (int, float))
        }
        candidate.damage_ratio_pred = float(model_result["stats"].get("damage_ratio", 0.0))
        candidate.model_score = (
            0.45 * float(candidate.model_metrics.get("f1_damaged", 0.0))
            + 0.35 * float(candidate.model_metrics.get("iou_damaged", 0.0))
            + 0.20 * float(candidate.model_metrics.get("mean_iou", 0.0))
        )
        candidate.final_score = 0.55 * candidate.visual_score + 0.45 * candidate.model_score
    return model_result


def candidate_to_row(candidate: Candidate, rank: int | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "rank": rank,
        "pair_id": candidate.pair_id,
        "split": candidate.split,
        "split_csv": str(candidate.split_csv),
        "disaster": candidate.disaster,
        "background_pixels": candidate.background_pixels,
        "no_damage_pixels": candidate.no_damage_pixels,
        "damaged_pixels": candidate.damaged_pixels,
        "building_pixels": candidate.building_pixels,
        "total_pixels": candidate.total_pixels,
        "building_ratio": candidate.building_ratio,
        "damaged_ratio_total": candidate.damaged_ratio_total,
        "damaged_ratio_building": candidate.damaged_ratio_building,
        "no_damage_ratio_building": candidate.no_damage_ratio_building,
        "pre_post_diff_building": candidate.pre_post_diff_building,
        "passes_filter": candidate.passes_filter,
        "visual_score": candidate.visual_score,
        "model_score": candidate.model_score,
        "final_score": candidate.final_score,
        "damage_ratio_pred": candidate.damage_ratio_pred,
    }
    if candidate.model_metrics:
        for key in ("f1_damaged", "iou_damaged", "mean_iou", "precision_damaged", "recall_damaged"):
            row[key] = candidate.model_metrics.get(key)
    return row


def write_ranked_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_recommended_ids(path: Path, selected: list[Candidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for candidate in selected:
            file.write(candidate.pair_id + "\n")


def resolve_device_name(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def write_readme(path: Path, selected: list[Candidate], args: argparse.Namespace, metadata: dict[str, Any]) -> None:
    lines = [
        "# Paires de démonstration Jalon 5",
        "",
        "Ce dossier contient des paires xBD sélectionnées automatiquement pour la démonstration Aftermath.",
        "Les images proviennent des splits existants du projet et sont redimensionnées à la taille configurée.",
        "",
        "## Critères de sélection",
        "",
        "- présence de bâtiments endommagés et non endommagés;",
        "- ratio bâtiment modéré;",
        "- équilibre visuel entre intact et endommagé;",
        "- diversité de catastrophes;",
        "- différence pré/post sur les zones bâtiment.",
        "",
        f"- `image_size` : {args.image_size}",
        f"- `run_model` : {bool(args.run_model)}",
        f"- `damage_tta` : {args.damage_tta}",
        f"- `building_threshold` : {args.building_threshold}",
        "",
        "## Top recommandé",
        "",
        "| Rang | pair_id | split | building ratio | damaged/building | score |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for rank, candidate in enumerate(selected, 1):
        lines.append(
            f"| {rank} | `{candidate.pair_id}` | {candidate.split} | "
            f"{candidate.building_ratio:.2%} | {candidate.damaged_ratio_building:.2%} | "
            f"{candidate.final_score:.3f} |"
        )
    if metadata.get("errors"):
        lines.extend(["", "## Avertissements", ""])
        lines.extend(f"- {error}" for error in metadata["errors"][:20])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def resize_with_letterbox(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS)
    canvas = Image.new("RGB", size, (15, 18, 25))
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def add_label(image: Image.Image, text: str) -> Image.Image:
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    draw.rectangle((0, 0, image.width, 28), fill=(0, 0, 0))
    draw.text((8, 6), text, fill=(255, 255, 255), font=font)
    return image


def make_contact_sheet(output_path: Path, selected: list[Candidate], args: argparse.Namespace) -> None:
    cell_w = args.contact_cell_width
    cell_h = int(cell_w * 0.72)
    labels_without_model = ["pre", "post", "target", "overlay_target"]
    labels_with_model = ["pre", "post", "target", "overlay_model", "final_damage"]
    columns = labels_with_model if args.run_model else labels_without_model

    rows = []
    for rank, candidate in enumerate(selected, 1):
        pair_dir = args.output_dir / f"{rank:02d}_{safe_filename(candidate.pair_id)}"
        panels = []
        for label in columns:
            path = pair_dir / f"{label}.png"
            if not path.exists():
                path = pair_dir / "overlay_target.png"
            with Image.open(path) as img:
                panel = resize_with_letterbox(img, (cell_w, cell_h))
            panels.append(add_label(panel, label))
        rows.append((candidate, panels))

    sheet_w = len(columns) * cell_w
    header_h = 34
    sheet_h = len(rows) * (cell_h + header_h)
    sheet = Image.new("RGB", (sheet_w, max(sheet_h, 1)), (7, 10, 16))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    y = 0
    for rank, (candidate, panels) in enumerate(rows, 1):
        draw.rectangle((0, y, sheet_w, y + header_h), fill=(21, 28, 40))
        title = (
            f"{rank:02d}  {candidate.pair_id}  | {candidate.split} | "
            f"building {candidate.building_ratio:.1%} | damaged/building {candidate.damaged_ratio_building:.1%}"
        )
        draw.text((10, y + 8), title, fill=(238, 242, 255), font=font)
        y += header_h
        x = 0
        for panel in panels:
            sheet.paste(panel, (x, y))
            x += cell_w
        y += cell_h
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def print_summary(selected: list[Candidate]) -> None:
    print()
    print("Top 12 paires de demo Jalon 5")
    print("=" * 32)
    for rank, candidate in enumerate(selected, 1):
        print(
            f"{rank:02d}. {candidate.pair_id} | split={candidate.split} | "
            f"building={candidate.building_ratio:.2%} | "
            f"damaged/building={candidate.damaged_ratio_building:.2%} | "
            f"score={candidate.final_score:.3f}"
        )
    print()
    print("Liste Python prete a coller dans RECOMMENDED_PAIR_IDS :")
    print("[")
    for candidate in selected:
        print(f'    "{candidate.pair_id}",')
    print("]")


def main() -> None:
    args = parse_args()
    split_specs = discover_splits(args)
    if not split_specs:
        raise SystemExit("No split CSV files found. Use --split-csv path/to/test_pairs.csv.")

    print("Scanning splits:")
    for spec in split_specs:
        print(f"- {spec.label}: {spec.path}")

    candidates, metadata = scan_candidates(args, split_specs)
    if not candidates:
        raise SystemExit("No candidates could be loaded.")

    selected = rank_candidates(candidates, args.top_k)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    app_state = None
    if args.run_model:
        print("Loading current app pipeline for model scoring...")
        app_state = load_app_pipeline(args)

    for rank, candidate in enumerate(selected, 1):
        export_candidate_assets(candidate, rank, args, app_state)

    selected_rows = [candidate_to_row(candidate, rank) for rank, candidate in enumerate(selected, 1)]
    all_ranked = sorted(candidates, key=lambda item: item.visual_score, reverse=True)
    all_rows = [candidate_to_row(candidate, rank) for rank, candidate in enumerate(all_ranked, 1)]
    write_ranked_csv(args.output_dir / "demo_pairs_ranked.csv", all_rows)
    write_ranked_csv(args.output_dir / "demo_pairs_selected.csv", selected_rows)
    write_recommended_ids(args.output_dir / "recommended_pair_ids.txt", selected)
    write_readme(args.output_dir / "README_DEMO_PAIRS.md", selected, args, metadata)
    with (args.output_dir / "selection_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False, default=str)
    make_contact_sheet(args.output_dir / "contact_sheet_top12.png", selected, args)

    print_summary(selected)
    print()
    print(f"Assets written to: {args.output_dir}")


if __name__ == "__main__":
    main()
