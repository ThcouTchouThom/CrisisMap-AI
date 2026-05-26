"""Train a binary building segmentation model for CrisisMap AI / Aftermath."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset


PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from crisismap.data.xbd_dataset import XBDDatasetError, XBDPairDataset  # noqa: E402
from crisismap.models.unet import UNet  # noqa: E402


MODEL_CHOICES = {"unet", "unetplusplus_effb3"}
INPUT_MODES = {"pre", "post", "pre-post"}
LOSS_CHOICES = {"bce-dice", "dice-bce", "focal-tversky"}
AUGMENT_MODES = {"none", "safe"}


class BuildingTrainingError(Exception):
    """Raised when building segmentation training cannot continue safely."""


class BuildingInputDataset(torch.utils.data.Dataset):
    """Select pre, post, or pre-post channels from XBDPairDataset samples."""

    def __init__(self, base_dataset: torch.utils.data.Dataset, input_mode: str) -> None:
        self.base_dataset = base_dataset
        self.input_mode = validate_input_mode(input_mode)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.base_dataset[index]
        image = sample["image"]
        if self.input_mode == "pre":
            image = image[:3]
        elif self.input_mode == "post":
            image = image[3:6]
        elif self.input_mode == "pre-post":
            image = image
        else:
            raise BuildingTrainingError(f"Unsupported input mode: {self.input_mode}")

        return {
            **sample,
            "image": image,
            "target": sample["target"],
        }


class BinaryDiceLoss(torch.nn.Module):
    """Binary Dice loss for logits and float masks."""

    def __init__(self, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(logits)
        intersection = torch.sum(probabilities * targets)
        denominator = torch.sum(probabilities) + torch.sum(targets)
        dice = (2.0 * intersection + self.epsilon) / (denominator + self.epsilon)
        return 1.0 - dice


class BCEDiceLoss(torch.nn.Module):
    """BCEWithLogits plus binary Dice loss."""

    def __init__(self) -> None:
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.dice = BinaryDiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce(logits, targets) + self.dice(logits, targets)


class FocalTverskyLoss(torch.nn.Module):
    """Binary focal Tversky loss for recall-oriented building segmentation."""

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 0.75,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(logits)
        true_positive = torch.sum(probabilities * targets)
        false_positive = torch.sum(probabilities * (1.0 - targets))
        false_negative = torch.sum((1.0 - probabilities) * targets)
        tversky = (true_positive + self.epsilon) / (
            true_positive
            + self.alpha * false_positive
            + self.beta * false_negative
            + self.epsilon
        )
        return torch.pow(1.0 - tversky, self.gamma)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a binary building segmentation model on xBD/xView2.",
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--val-csv", required=True, type=Path)
    parser.add_argument("--test-csv", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", choices=sorted(MODEL_CHOICES), default="unetplusplus_effb3")
    parser.add_argument("--input-mode", choices=sorted(INPUT_MODES), default="pre")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--loss", choices=sorted(LOSS_CHOICES), default="focal-tversky")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision.")
    parser.add_argument(
        "--augment-mode",
        choices=sorted(AUGMENT_MODES),
        default="safe",
        help="Train-only augmentation mode.",
    )
    parser.add_argument(
        "--target-mode",
        choices=["building-binary"],
        default="building-binary",
        help="Only building-binary is supported by this script.",
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--focal-tversky-alpha", type=float, default=0.3)
    parser.add_argument("--focal-tversky-beta", type=float, default=0.7)
    parser.add_argument("--focal-tversky-gamma", type=float, default=0.75)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Print concise train progress every N batches. Use 0 to disable.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in ["image_size", "batch_size", "epochs"]:
        if getattr(args, name) <= 0:
            raise BuildingTrainingError(f"--{name.replace('_', '-')} must be positive.")
    if args.lr <= 0:
        raise BuildingTrainingError("--lr must be positive.")
    if args.num_workers < 0:
        raise BuildingTrainingError("--num-workers must be non-negative.")
    if args.log_every < 0:
        raise BuildingTrainingError("--log-every must be non-negative.")
    for name in ["max_train_samples", "max_val_samples", "max_test_samples"]:
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise BuildingTrainingError(f"--{name.replace('_', '-')} must be positive.")
    for name in ["focal_tversky_alpha", "focal_tversky_beta", "focal_tversky_gamma"]:
        value = getattr(args, name)
        if value <= 0:
            raise BuildingTrainingError(f"--{name.replace('_', '-')} must be positive.")


def validate_input_mode(input_mode: str) -> str:
    if input_mode not in INPUT_MODES:
        raise BuildingTrainingError(
            "input_mode must be one of: " + ", ".join(sorted(INPUT_MODES))
        )
    return input_mode


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise BuildingTrainingError("CUDA was requested, but CUDA is not available.")
    return device


def prepare_output_dir(path: Path) -> Path:
    output_dir = path.expanduser().resolve()
    if "raw" in {part.lower() for part in output_dir.parts}:
        raise BuildingTrainingError("Refusing to write outputs inside a raw data folder.")
    output_dir.mkdir(parents=True, exist_ok=True)
    if not output_dir.is_dir():
        raise BuildingTrainingError(f"Output path is not a directory: {output_dir}")
    return output_dir


def make_dataset(
    root: Path,
    split_csv: Path,
    image_size: int,
    input_mode: str,
    target_mode: str,
    max_samples: int | None,
    augment_mode: str,
) -> torch.utils.data.Dataset:
    base_dataset = XBDPairDataset(
        root=root,
        split_csv=split_csv,
        image_size=image_size,
        target_mode=target_mode,
        augment_mode=augment_mode,
        augment_prob=0.5 if augment_mode != "none" else 0.0,
    )
    if max_samples is not None:
        base_dataset = Subset(base_dataset, range(min(max_samples, len(base_dataset))))
    return BuildingInputDataset(base_dataset, input_mode)


def make_loader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def input_channels(input_mode: str) -> int:
    return 6 if input_mode == "pre-post" else 3


def build_model(
    model_name: str,
    in_channels: int,
    device: torch.device,
) -> tuple[torch.nn.Module, str]:
    if model_name == "unet":
        return UNet(in_channels=in_channels, num_classes=1, base_channels=32).to(device), "unet"

    if model_name == "unetplusplus_effb3":
        try:
            import segmentation_models_pytorch as smp  # type: ignore

            model = smp.UnetPlusPlus(
                encoder_name="efficientnet-b3",
                encoder_weights=None,
                in_channels=in_channels,
                classes=1,
                activation=None,
            )
            return model.to(device), "unetplusplus_effb3"
        except ImportError:
            print(
                "WARNING: segmentation_models_pytorch is not installed; "
                "falling back to the local UNet."
            )
        except Exception as exc:
            print(
                "WARNING: could not create U-Net++ EfficientNet-B3 model "
                f"({exc}); falling back to the local UNet."
            )
        return UNet(in_channels=in_channels, num_classes=1, base_channels=32).to(device), "unet_fallback"

    raise BuildingTrainingError(f"Unsupported model: {model_name}")


def build_loss(args: argparse.Namespace) -> torch.nn.Module:
    if args.loss in {"bce-dice", "dice-bce"}:
        return BCEDiceLoss()
    if args.loss == "focal-tversky":
        return FocalTverskyLoss(
            alpha=args.focal_tversky_alpha,
            beta=args.focal_tversky_beta,
            gamma=args.focal_tversky_gamma,
        )
    raise BuildingTrainingError(f"Unsupported loss: {args.loss}")


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def normalize_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 3:
        return logits.unsqueeze(1)
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise BuildingTrainingError(
            f"Expected binary logits with shape Bx1xHxW, got {tuple(logits.shape)}."
        )
    return logits


def targets_to_float(targets: torch.Tensor) -> torch.Tensor:
    return targets.to(dtype=torch.float32).unsqueeze(1)


def autocast_context(use_amp: bool):
    if use_amp:
        return torch.cuda.amp.autocast()
    return nullcontext()


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    epoch: int,
    log_every: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    total_batches = len(loader)
    epoch_started_at = time.time()

    for batch_index, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        target_float = targets_to_float(targets)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(use_amp):
            logits = normalize_logits(model(images))
            loss = criterion(logits, target_float)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.shape[0]
        total_loss += float(loss.detach().item()) * batch_size
        total_samples += batch_size

        should_log = (
            log_every > 0
            and (
                batch_index == 1
                or batch_index % log_every == 0
                or batch_index == total_batches
            )
        )
        if should_log:
            elapsed_minutes = (time.time() - epoch_started_at) / 60.0
            running_loss = total_loss / max(total_samples, 1)
            print(
                f"Epoch {epoch:03d} batch {batch_index}/{total_batches} | "
                f"loss {float(loss.detach().item()):.4f} | "
                f"running {running_loss:.4f} | "
                f"elapsed {elapsed_minutes:.1f} min",
                flush=True,
            )
    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    use_amp: bool,
) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        target_float = targets_to_float(targets)

        with autocast_context(use_amp):
            logits = normalize_logits(model(images))
            loss = criterion(logits, target_float)

        probabilities = torch.sigmoid(logits).squeeze(1)
        predictions = probabilities >= 0.5
        target_bool = targets > 0

        counts["tp"] += int(torch.count_nonzero(predictions & target_bool).item())
        counts["tn"] += int(torch.count_nonzero((~predictions) & (~target_bool)).item())
        counts["fp"] += int(torch.count_nonzero(predictions & (~target_bool)).item())
        counts["fn"] += int(torch.count_nonzero((~predictions) & target_bool).item())

        batch_size = images.shape[0]
        total_loss += float(loss.detach().item()) * batch_size
        total_samples += batch_size

    metrics = metrics_from_counts(counts)
    metrics["loss"] = total_loss / max(total_samples, 1)
    metrics["confusion_counts"] = counts
    return metrics


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def metrics_from_counts(counts: dict[str, int]) -> dict[str, float]:
    tp = float(counts["tp"])
    tn = float(counts["tn"])
    fp = float(counts["fp"])
    fn = float(counts["fn"])
    total = tp + tn + fp + fn

    background_iou = safe_divide(tn, tn + fp + fn)
    building_iou = safe_divide(tp, tp + fp + fn)
    building_precision = safe_divide(tp, tp + fp)
    building_recall = safe_divide(tp, tp + fn)
    building_f1 = safe_divide(
        2.0 * building_precision * building_recall,
        building_precision + building_recall,
    )

    return {
        "pixel_accuracy": safe_divide(tp + tn, total),
        "mean_iou": (background_iou + building_iou) / 2.0,
        "background_iou": background_iou,
        "building_iou": building_iou,
        "building_precision": building_precision,
        "building_recall": building_recall,
        "building_f1": building_f1,
    }


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, object],
    args: argparse.Namespace,
    actual_model: str,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": vars(args),
        "actual_model": actual_model,
    }
    torch.save(checkpoint, path)


def load_checkpoint_file(path: Path, device: torch.device) -> object:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_resume_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> int:
    if not path.exists():
        raise BuildingTrainingError(f"Resume checkpoint does not exist: {path}")
    checkpoint = load_checkpoint_file(path, device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        epoch = int(checkpoint.get("epoch", 0))
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        state_dict = checkpoint
        epoch = 0
    model.load_state_dict(clean_state_dict(state_dict))
    return epoch + 1


def clean_state_dict(state_dict: object) -> dict[str, torch.Tensor]:
    if not isinstance(state_dict, dict):
        raise BuildingTrainingError("Checkpoint does not contain a state_dict.")
    cleaned = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            cleaned[str(key).removeprefix("module.")] = value
    if not cleaned:
        raise BuildingTrainingError("Checkpoint contains no tensor weights.")
    return cleaned


def write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=str)


def write_summary_csv(
    path: Path,
    args: argparse.Namespace,
    actual_model: str,
    best_val_metrics: dict[str, object],
    test_metrics: dict[str, object] | None,
) -> None:
    fields = [
        "experiment",
        "model_requested",
        "model_actual",
        "input_mode",
        "image_size",
        "batch_size",
        "epochs",
        "lr",
        "loss",
        "val_building_iou",
        "val_building_f1",
        "test_building_iou",
        "test_building_f1",
        "test_mean_iou",
    ]
    row = {
        "experiment": args.output_dir.name,
        "model_requested": args.model,
        "model_actual": actual_model,
        "input_mode": args.input_mode,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "loss": args.loss,
        "val_building_iou": best_val_metrics.get("building_iou"),
        "val_building_f1": best_val_metrics.get("building_f1"),
        "test_building_iou": None if test_metrics is None else test_metrics.get("building_iou"),
        "test_building_f1": None if test_metrics is None else test_metrics.get("building_f1"),
        "test_mean_iou": None if test_metrics is None else test_metrics.get("mean_iou"),
    }
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)


def metric_or_default(path: Path, metric: str, default: float) -> float:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        return float(payload.get(metric, default))
    except (OSError, ValueError, json.JSONDecodeError, TypeError):
        return default


def print_epoch(epoch: int, train_loss: float, val_metrics: dict[str, object]) -> None:
    print(
        f"Epoch {epoch:03d} | "
        f"train loss {train_loss:.4f} | "
        f"val loss {val_metrics['loss']:.4f} | "
        f"pixel acc {val_metrics['pixel_accuracy']:.4f} | "
        f"mean IoU {val_metrics['mean_iou']:.4f} | "
        f"building IoU {val_metrics['building_iou']:.4f} | "
        f"building F1 {val_metrics['building_f1']:.4f}"
    )


def train(args: argparse.Namespace) -> None:
    validate_args(args)
    output_dir = prepare_output_dir(args.output_dir)
    device = resolve_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")

    train_dataset = make_dataset(
        args.root,
        args.train_csv,
        args.image_size,
        args.input_mode,
        args.target_mode,
        args.max_train_samples,
        augment_mode=args.augment_mode,
    )
    val_dataset = make_dataset(
        args.root,
        args.val_csv,
        args.image_size,
        args.input_mode,
        args.target_mode,
        args.max_val_samples,
        augment_mode="none",
    )
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise BuildingTrainingError("Train and validation datasets must be non-empty.")

    train_loader = make_loader(
        train_dataset,
        args.batch_size,
        args.num_workers,
        shuffle=True,
        device=device,
    )
    val_loader = make_loader(
        val_dataset,
        args.batch_size,
        args.num_workers,
        shuffle=False,
        device=device,
    )

    model, actual_model = build_model(args.model, input_channels(args.input_mode), device)
    criterion = build_loss(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch = 1
    if args.resume is not None:
        start_epoch = load_resume_checkpoint(args.resume, model, optimizer, device)

    best_metric = metric_or_default(output_dir / "best_val_metrics.json", "building_iou", -1.0)
    history: list[dict[str, object]] = []
    history_path = output_dir / "metrics_history.json"
    if history_path.exists():
        try:
            with history_path.open("r", encoding="utf-8") as file:
                existing_history = json.load(file)
            if isinstance(existing_history, list):
                history = existing_history
        except (OSError, json.JSONDecodeError):
            history = []

    print("CrisisMap AI - Building Segmentation Training")
    print("=" * 49)
    print(f"Device: {device}")
    print(f"AMP: {'enabled' if use_amp else 'disabled'}")
    print(f"Model requested: {args.model}")
    print(f"Model actual: {actual_model}")
    print(f"Input mode: {args.input_mode}")
    print(f"Target mode: {args.target_mode}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Parameters: {parameter_count(model):,}")
    print(f"Loss: {args.loss}")
    print(f"Augment mode: {args.augment_mode}")
    print(f"Log every: {args.log_every} batches")

    started_at = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            use_amp,
            epoch,
            args.log_every,
        )
        val_metrics = evaluate(model, val_loader, criterion, device, use_amp)
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_pixel_accuracy": val_metrics["pixel_accuracy"],
            "val_mean_iou": val_metrics["mean_iou"],
            "val_background_iou": val_metrics["background_iou"],
            "val_building_iou": val_metrics["building_iou"],
            "val_building_precision": val_metrics["building_precision"],
            "val_building_recall": val_metrics["building_recall"],
            "val_building_f1": val_metrics["building_f1"],
            "epoch_seconds": time.time() - epoch_start,
        }
        history.append(epoch_metrics)
        print_epoch(epoch, train_loss, val_metrics)

        if float(val_metrics["building_iou"]) > best_metric:
            best_metric = float(val_metrics["building_iou"])
            save_checkpoint(
                output_dir / "best_building.pt",
                model,
                optimizer,
                epoch,
                val_metrics,
                args,
                actual_model,
            )
            write_json(output_dir / "best_val_metrics.json", val_metrics)

        save_checkpoint(
            output_dir / "last_building.pt",
            model,
            optimizer,
            epoch,
            val_metrics,
            args,
            actual_model,
        )
        write_json(history_path, history)

    best_val_metrics = read_metrics(output_dir / "best_val_metrics.json")
    test_metrics = None
    if args.test_csv is not None:
        test_dataset = make_dataset(
            args.root,
            args.test_csv,
            args.image_size,
            args.input_mode,
            args.target_mode,
            args.max_test_samples,
            augment_mode="none",
        )
        test_loader = make_loader(
            test_dataset,
            args.batch_size,
            args.num_workers,
            shuffle=False,
            device=device,
        )
        load_resume_checkpoint(output_dir / "best_building.pt", model, None, device)
        test_metrics = evaluate(model, test_loader, criterion, device, use_amp)
        write_json(output_dir / "test_metrics.json", test_metrics)

    write_summary_csv(
        output_dir / "summary.csv",
        args,
        actual_model,
        best_val_metrics,
        test_metrics,
    )
    print(f"Training complete in {(time.time() - started_at) / 60.0:.2f} minutes.")
    print(f"Best validation building IoU: {best_metric:.4f}")
    if test_metrics is not None:
        print(f"Test building IoU: {test_metrics['building_iou']:.4f}")
        print(f"Test building F1: {test_metrics['building_f1']:.4f}")
    print(f"Outputs saved to: {output_dir}")


def read_metrics(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return payload if isinstance(payload, dict) else {}


def main() -> int:
    args = parse_args()
    try:
        train(args)
    except (
        BuildingTrainingError,
        XBDDatasetError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
