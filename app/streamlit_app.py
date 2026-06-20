"""Aftermath - prototype Streamlit de démonstration Jalon 5."""

from __future__ import annotations

import base64
import io
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import torch
from PIL import Image, ImageOps

RESAMPLE_BILINEAR = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
RESAMPLE_NEAREST = Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST


# == Chemins du projet =========================================================


def find_project_root(start: Path) -> Path:
    """Trouve la racine du dépôt en cherchant src/ et scripts/."""
    for candidate in (start, *start.parents):
        if (candidate / "src").is_dir() and (candidate / "scripts").is_dir():
            return candidate
    return start


PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)
PROJECT_SRC = PROJECT_ROOT / "src"
PROJECT_SCRIPTS = PROJECT_ROOT / "scripts"

for path in (PROJECT_SRC, PROJECT_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from crisismap.data.xbd_dataset import XBDDatasetError, XBDPairDataset as XBDDataset  # noqa: E402
from crisismap.models.damage_model_factory import (  # noqa: E402
    DamageModelFactoryError,
    create_damage_model,
)

try:
    from train_building_segmentation import (  # noqa: E402
        BuildingTrainingError,
        build_model as build_building_model,
        clean_state_dict as clean_building_state_dict,
        input_channels as building_input_channels,
        normalize_logits as normalize_building_logits,
    )

    BUILDING_MODULE_AVAILABLE = True
except ImportError:
    BUILDING_MODULE_AVAILABLE = False
    BuildingTrainingError = Exception

try:
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# == Constantes ================================================================


DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "xbd" / "train"
SAMPLE_PAIRS_ROOT = PROJECT_ROOT / "sample_data" / "demo_pairs"
SPLIT_DIR_CANDIDATES = [
    PROJECT_ROOT / "data" / "processed" / "splits",
    PROJECT_ROOT / "data" / "processed" / "splits_full",
]
TARGET_MODE = "3-class"


def first_existing_path(paths: list[Path]) -> Path:
    """Retourne le premier chemin existant, sinon le premier candidat."""
    return next((path for path in paths if path.exists()), paths[0])


DAMAGE_CHAMPION_CHECKPOINT_CANDIDATES = [
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
    / "dftv2_hist1000_attention_sqrt2_ft_250_seed0"
    / "best_damage_arch_portable.pt",
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
    / "dftv2_hist1000_attention_sqrt2_ft_250_seed0"
    / "best_damage_arch.pt",
]

DAMAGE_MODELS: dict[str, dict[str, Any]] = {
    "damage_champion_v2": {
        "label": "Damage champion v2 - Siamese Attention",
        "checkpoint": first_existing_path(DAMAGE_CHAMPION_CHECKPOINT_CANDIDATES),
        "checkpoint_candidates": DAMAGE_CHAMPION_CHECKPOINT_CANDIDATES,
        "model_name": "siamese_unet_attention",
        "description": (
            "Meilleur modèle damage actuel : Siamese Attention, focal-Tversky, "
            "hist1000, sampler sqrt2, 250 epochs."
        ),
        "in_channels": 6,
        "base_channels": 32,
        "image_size": 1024,
        "summary": "F1 damaged = 0.7013, IoU damaged = 0.5400, mean IoU = 0.7283",
    },
    "unet_long250": {
        "label": "U-Net long250 - baseline forte",
        "checkpoint": PROJECT_ROOT
        / "outputs"
        / "checkpoints"
        / "unet_1024_long250_noleak_match_hist_all_aug-safe_sampler-damage-sqrt-alpha4_250epochs"
        / "best_unet_portable.pt",
        "model_name": "unet",
        "description": "Baseline U-Net forte, 1024 px, no-leak, 250 epochs.",
        "in_channels": 6,
        "base_channels": 32,
        "image_size": 1024,
        "summary": "Avec TTA d4 : F1 damaged = 0.6313, IoU damaged = 0.4612",
    },
    "unet_baseline": {
        "label": "U-Net baseline",
        "checkpoint": PROJECT_ROOT
        / "outputs"
        / "checkpoints"
        / "unet_baseline"
        / "best_unet_portable.pt",
        "model_name": "unet",
        "description": "Ancien U-Net de référence, utile seulement comme fallback.",
        "in_channels": 6,
        "base_channels": 32,
        "image_size": 512,
        "summary": "Fallback rapide si les checkpoints 1024 ne sont pas disponibles.",
    },
}

BUILDING_CHECKPOINT_CANDIDATES = [
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
    / "b400_effb4_sampler8_ft"
    / "best_building_portable.pt",
    PROJECT_ROOT / "outputs" / "checkpoints" / "b400_effb4_sampler8_ft" / "best_building.pt",
]
BUILDING_CHECKPOINT = first_existing_path(BUILDING_CHECKPOINT_CANDIDATES)
BUILDING_MODEL_NAME = "unetplusplus_effb4"
BUILDING_INPUT_MODE = "pre"
BUILDING_SUMMARY = "Building b400 : F1 building = 0.8504, IoU building = 0.7398"

PIPELINES: dict[str, dict[str, Any]] = {
    "damage_only": {
        "label": "Rapide : damage seul",
        "damage_tta": "none",
        "use_building": False,
        "description": "Inférence damage brute, la plus rapide.",
    },
    "damage_tta": {
        "label": "Qualité : damage + TTA d4",
        "damage_tta": "d4",
        "use_building": False,
        "description": "Moyenne D4 sur le modèle damage, plus stable mais plus lente.",
    },
    "damage_tta_building": {
        "label": "Qualité maximale : damage + TTA d4 + building post-process",
        "damage_tta": "d4",
        "use_building": True,
        "description": "Pipeline recommandé : TTA damage, masque bâtiment, component majority.",
    },
}

SOURCE_MODES = {
    "upload": "Téléverser des images",
    "embedded": "Exemples inclus",
    "dataset": "Exemples du dataset",
}

THEME_LABELS = ["Sombre", "Clair", "Système"]

RECOMMENDED_PAIR_IDS = [
    "hurricane-michael_00000085",
    "hurricane-michael_00000446",
    "palu-tsunami_00000019",
    "palu-tsunami_00000183",
    "santa-rosa-wildfire_00000117",
    "santa-rosa-wildfire_00000011",
    "hurricane-michael_00000239",
    "hurricane-harvey_00000478",
]

DISASTER_INFO: dict[str, tuple[str, str]] = {
    "hurricane-harvey": ("Houston, TX", "#3b82f6"),
    "hurricane-michael": ("Panama City, FL", "#3b82f6"),
    "hurricane-florence": ("Wilmington, NC", "#3b82f6"),
    "santa-rosa-wildfire": ("Santa Rosa, CA", "#f97316"),
    "socal-fire": ("Los Angeles, CA", "#f97316"),
    "woolsey-fire": ("Malibu, CA", "#f97316"),
    "palu-tsunami": ("Palu, Indonésie", "#06b6d4"),
    "sunda-tsunami": ("Indonésie", "#06b6d4"),
    "midwest-flooding": ("Iowa, USA", "#6366f1"),
    "joplin-tornado": ("Joplin, MO", "#a855f7"),
    "guatemala-volcano": ("Guatemala", "#ef4444"),
    "mexico-earthquake": ("Mexico City", "#d97706"),
}

CLASS_COLORS = {
    0: np.array([0, 0, 0], dtype=np.uint8),
    1: np.array([0, 170, 80], dtype=np.uint8),
    2: np.array([220, 40, 40], dtype=np.uint8),
}


# == Style =====================================================================


def theme_colors(theme: str) -> dict[str, str]:
    dark = theme == "Sombre"
    if dark:
        return {
            "bg": "#0a0d12",
            "surface": "#111620",
            "surface2": "#181e2a",
            "border": "#222c3c",
            "text_hi": "#eaf0fb",
            "text_md": "#93a4bb",
            "text_lo": "#69778b",
            "accent": "#dc2828",
        }
    return {
        "bg": "#f4f6fa",
        "surface": "#ffffff",
        "surface2": "#eef2f7",
        "border": "#d1dae9",
        "text_hi": "#0f172a",
        "text_md": "#475569",
        "text_lo": "#94a3b8",
        "accent": "#dc2828",
    }


def build_css(theme: str) -> str:
    colors = theme_colors("Sombre" if theme == "Sombre" else "Clair")
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, .stApp {{
    font-family: 'Inter', sans-serif !important;
    background-color: {colors["bg"]} !important;
    color: {colors["text_hi"]} !important;
}}
[data-testid="stSidebar"] {{
    background-color: {colors["surface"]} !important;
    border-right: 1px solid {colors["border"]} !important;
}}
.stButton > button {{
    background: {colors["accent"]} !important;
    color: white !important;
    border: none !important;
    border-radius: 7px !important;
    font-weight: 600 !important;
}}
.stDownloadButton > button {{
    background: {colors["surface2"]} !important;
    color: {colors["text_hi"]} !important;
    border: 1px solid {colors["border"]} !important;
    border-radius: 7px !important;
}}
.stTabs [data-baseweb="tab-list"] {{
    border-bottom: 1px solid {colors["border"]} !important;
}}
.stTabs [data-baseweb="tab"] {{
    font-size: 0.86rem !important;
    color: {colors["text_md"]} !important;
}}
.stTabs [aria-selected="true"] {{
    color: {colors["accent"]} !important;
    border-bottom: 2px solid {colors["accent"]} !important;
}}
.amr-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1rem;
    border-bottom: 1px solid {colors["border"]};
    padding: 1.1rem 0 1rem;
    margin-bottom: 1.4rem;
}}
.amr-logo {{
    font-family: 'Syne', sans-serif;
    font-size: 1.85rem;
    font-weight: 700;
    color: {colors["accent"]};
    letter-spacing: -0.02em;
}}
.amr-tagline {{
    color: {colors["text_lo"]};
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
}}
.amr-badges {{
    display: flex;
    gap: 0.45rem;
    flex-wrap: wrap;
    justify-content: flex-end;
}}
.amr-badge {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.67rem;
    color: {colors["text_md"]};
    border: 1px solid {colors["border"]};
    background: {colors["surface"]};
    padding: 0.22rem 0.58rem;
    border-radius: 999px;
}}
.amr-badge.accent {{
    color: {colors["accent"]};
    border-color: {colors["accent"]};
}}
.section-lbl {{
    display: flex;
    align-items: center;
    gap: 0.65rem;
    margin: 1.2rem 0 0.65rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.66rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: {colors["text_lo"]};
}}
.section-lbl::after {{
    content: '';
    flex: 1;
    height: 1px;
    background: {colors["border"]};
}}
.legend-bar {{
    display: flex;
    gap: 1.25rem;
    flex-wrap: wrap;
    padding: 0.7rem 1rem;
    background: {colors["surface"]};
    border: 1px solid {colors["border"]};
    border-radius: 8px;
    margin-bottom: 1rem;
}}
.leg-item {{
    display: flex;
    align-items: center;
    gap: 0.42rem;
    color: {colors["text_md"]};
    font-size: 0.84rem;
}}
.leg-dot {{
    width: 10px;
    height: 10px;
    border-radius: 2px;
}}
.metric-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(145px, 1fr));
    gap: 0.7rem;
    margin: 0.5rem 0 1rem;
}}
.mcard {{
    border: 1px solid {colors["border"]};
    background: {colors["surface"]};
    border-radius: 8px;
    padding: 0.85rem 0.9rem;
    text-align: center;
}}
.mval {{
    font-family: 'Syne', sans-serif;
    color: {colors["text_hi"]};
    font-size: 1.35rem;
    font-weight: 700;
}}
.mlbl {{
    color: {colors["text_lo"]};
    font-size: 0.65rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}}
.info-strip {{
    border-left: 3px solid {colors["accent"]};
    background: rgba(220, 40, 40, 0.08);
    border-radius: 0 8px 8px 0;
    padding: 0.65rem 0.9rem;
    margin: 0.5rem 0 1rem;
    color: {colors["text_md"]};
}}
.history-row {{
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    border: 1px solid {colors["border"]};
    background: {colors["surface"]};
    border-radius: 8px;
    padding: 0.7rem 0.9rem;
    margin-bottom: 0.45rem;
}}
.hr-id {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.76rem;
}}
.hr-meta {{
    color: {colors["text_lo"]};
    font-size: 0.72rem;
}}
.hr-dmg {{
    color: {colors["accent"]};
    font-family: 'Syne', sans-serif;
    font-weight: 700;
}}
#MainMenu, footer {{
    visibility: hidden !important;
}}
</style>
"""


# == Utilitaires ===============================================================


class AppError(Exception):
    """Erreur applicative affichable dans Streamlit."""


def rel_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def img_to_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def tensor_to_rgb(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)


def read_upload(file, size: int) -> np.ndarray:
    with Image.open(file) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        img = img.resize((size, size), RESAMPLE_BILINEAR)
        return np.asarray(img, dtype=np.uint8)


def read_rgb_path(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        img = img.resize((size, size), RESAMPLE_BILINEAR)
        return np.asarray(img, dtype=np.uint8)


def read_demo_target(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = img.resize((size, size), RESAMPLE_NEAREST)
        rgb = np.asarray(img, dtype=np.uint8)

    red = (rgb[:, :, 0] > 140) & (rgb[:, :, 1] < 120)
    green = (rgb[:, :, 1] > 120) & (rgb[:, :, 0] < 120)
    target = np.zeros(rgb.shape[:2], dtype=np.int16)
    target[green] = 1
    target[red] = 2
    return target


def colorize(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        out[mask == class_id] = color
    return out


def colorize_building(mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return np.zeros((32, 32, 3), dtype=np.uint8)
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    out[mask.astype(bool)] = np.array([0, 180, 220], dtype=np.uint8)
    return out


def make_overlay(post: np.ndarray, pred: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    colors = colorize(pred).astype(np.float32)
    opacity = np.zeros(pred.shape, dtype=np.float32)
    opacity[pred == 1] = 0.35
    opacity[pred == 2] = alpha
    opacity = opacity[:, :, None]
    return np.clip(post.astype(np.float32) * (1 - opacity) + colors * opacity, 0, 255).astype(np.uint8)


def entropy_map(probs: np.ndarray) -> np.ndarray:
    eps = 1e-8
    entropy = -np.sum(probs * np.log(probs + eps), axis=-1) / np.log(probs.shape[-1])
    try:
        import matplotlib.cm as mcm

        return (mcm.plasma(entropy)[:, :, :3] * 255).astype(np.uint8)
    except ImportError:
        gray = (entropy * 255).astype(np.uint8)
        return np.stack([gray, gray, gray], axis=-1)


def prediction_stats(pred: np.ndarray) -> dict[str, float | int]:
    building_pixels = int(np.isin(pred, [1, 2]).sum())
    damaged_pixels = int((pred == 2).sum())
    return {
        "building_pixels": building_pixels,
        "damaged_pixels": damaged_pixels,
        "damage_ratio": damaged_pixels / building_pixels if building_pixels else 0.0,
    }


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    confusion = np.zeros((3, 3), dtype=np.float64)
    valid = (target >= 0) & (target < 3)
    idx = target[valid].astype(int) * 3 + pred[valid].astype(int)
    confusion += np.bincount(idx, minlength=9).reshape(3, 3)
    tp = np.diag(confusion)
    total = confusion.sum()
    unions = confusion.sum(1) + confusion.sum(0) - tp
    iou = np.divide(tp, unions, out=np.full(3, np.nan), where=unions > 0)

    damaged_tp = confusion[2, 2]
    damaged_pred = confusion[:, 2].sum()
    damaged_true = confusion[2, :].sum()
    precision = damaged_tp / damaged_pred if damaged_pred else 0.0
    recall = damaged_tp / damaged_true if damaged_true else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    metrics: dict[str, float | int] = {
        "pixel_accuracy": float(tp.sum() / total) if total else 0.0,
        "mean_iou": float(np.nanmean(iou)),
        "iou_background": 0.0 if np.isnan(iou[0]) else float(iou[0]),
        "iou_no_damage": 0.0 if np.isnan(iou[1]) else float(iou[1]),
        "iou_damaged": 0.0 if np.isnan(iou[2]) else float(iou[2]),
        "precision_damaged": float(precision),
        "recall_damaged": float(recall),
        "f1_damaged": float(f1),
    }
    metrics.update(prediction_stats(pred))
    return metrics


def split_csv_path(split: str) -> Path:
    for split_dir in SPLIT_DIR_CANDIDATES:
        candidate = split_dir / f"{split}_pairs.csv"
        if candidate.exists():
            return candidate
    return SPLIT_DIR_CANDIDATES[0] / f"{split}_pairs.csv"


def load_checkpoint(path: Path, device: torch.device) -> object:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def clean_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict", checkpoint)
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, dict):
        raise AppError("Le checkpoint ne contient pas de state_dict valide.")
    cleaned = {
        str(key).removeprefix("module."): value
        for key, value in state_dict.items()
        if isinstance(value, torch.Tensor)
    }
    if not cleaned:
        raise AppError("Le checkpoint ne contient aucun tenseur de poids.")
    return cleaned


def get_disaster(pair_id: str) -> tuple[str, str, str] | None:
    for prefix, (place, color) in DISASTER_INFO.items():
        if pair_id.startswith(prefix):
            return prefix, place, color
    return None


# == Chargement des données et modèles ========================================


@st.cache_data(show_spinner=False)
def load_split(split: str) -> pd.DataFrame:
    path = split_csv_path(split)
    if not path.exists():
        raise AppError(f"Split introuvable : {path}")
    df = pd.read_csv(path)
    if "pair_id" not in df.columns or df.empty:
        raise AppError(f"Split invalide ou vide : {path}")
    return df


@st.cache_resource(show_spinner="Chargement du modèle damage...")
def load_damage_model(
    checkpoint: str,
    model_name: str,
    in_channels: int,
    base_channels: int,
    device_name: str,
):
    path = Path(checkpoint)
    if not path.exists():
        raise AppError(f"Checkpoint damage manquant : {path}")
    device = torch.device(device_name)
    try:
        model = create_damage_model(
            model_name,
            num_classes=3,
            in_channels=in_channels,
            base_channels=base_channels,
        ).to(device)
    except DamageModelFactoryError as exc:
        raise AppError(f"Modèle damage non supporté : {model_name}. {exc}") from exc

    checkpoint_obj = load_checkpoint(path, device)
    try:
        model.load_state_dict(clean_state_dict(checkpoint_obj))
    except RuntimeError as exc:
        raise AppError(
            "Checkpoint damage incompatible avec le modèle demandé. "
            f"Modèle : {model_name}. Checkpoint : {path}"
        ) from exc
    model.eval()
    return model


@st.cache_resource(show_spinner="Chargement du modèle bâtiment...")
def load_building_model(device_name: str):
    if not BUILDING_MODULE_AVAILABLE:
        raise AppError("Le module building n'est pas disponible : train_building_segmentation.py est introuvable.")
    if not BUILDING_CHECKPOINT.exists():
        choices = "\n".join(f"- {path}" for path in BUILDING_CHECKPOINT_CANDIDATES)
        raise AppError(f"Checkpoint building manquant. Chemins testés :\n{choices}")

    device = torch.device(device_name)
    try:
        model, _ = build_building_model(
            BUILDING_MODEL_NAME,
            building_input_channels(BUILDING_INPUT_MODE),
            device,
        )
        checkpoint_obj = load_checkpoint(BUILDING_CHECKPOINT, device)
        state_dict = checkpoint_obj.get("model_state_dict", checkpoint_obj) if isinstance(checkpoint_obj, dict) else checkpoint_obj
        model.load_state_dict(clean_building_state_dict(state_dict))
    except (RuntimeError, BuildingTrainingError, OSError, ValueError) as exc:
        raise AppError(
            "Checkpoint building incompatible ou modèle building indisponible. "
            f"Modèle : {BUILDING_MODEL_NAME}. Checkpoint : {BUILDING_CHECKPOINT}"
        ) from exc
    model.eval()
    return model


def load_sample(split: str, pair_id: str, image_size: int) -> dict[str, Any]:
    dataset = XBDDataset(
        root=DATA_ROOT,
        split_csv=split_csv_path(split),
        image_size=image_size,
        target_mode=TARGET_MODE,
    )
    matches = dataset.samples.index[dataset.samples["pair_id"].astype(str) == pair_id].tolist()
    if not matches:
        raise AppError(f"Paire introuvable dans le split {split} : {pair_id}")
    return dataset[int(matches[0])]


# == Inférence =================================================================


@torch.no_grad()
def tta_d4_logits(model, batch: torch.Tensor) -> torch.Tensor:
    total = None
    for k in range(4):
        for flip in (False, True):
            view = torch.rot90(batch, k=k, dims=(-2, -1))
            if flip:
                view = torch.flip(view, dims=(-1,))
            logits = model(view).float()
            if flip:
                logits = torch.flip(logits, dims=(-1,))
            logits = torch.rot90(logits, k=-k, dims=(-2, -1))
            total = logits if total is None else total + logits
    return total / 8.0


def connected_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    try:
        from scipy import ndimage

        return ndimage.label(mask.astype(bool), structure=np.ones((3, 3), dtype=np.uint8))
    except Exception:
        pass

    try:
        from skimage.measure import label as sk_label

        labels = sk_label(mask.astype(bool), connectivity=2).astype(np.int32)
        return labels, int(labels.max())
    except Exception:
        pass

    height, width = mask.shape
    labels = np.zeros((height, width), dtype=np.int32)
    current = 0
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or labels[y, x] != 0:
                continue
            current += 1
            labels[y, x] = current
            stack = [(y, x)]
            while stack:
                cy, cx = stack.pop()
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = current
                            stack.append((ny, nx))
    return labels, current


def component_majority(raw_pred: np.ndarray, building_mask: np.ndarray) -> np.ndarray:
    labels, n_components = connected_components(building_mask)
    final_pred = np.zeros_like(raw_pred, dtype=np.int16)
    for component_id in range(1, n_components + 1):
        component = labels == component_id
        values = raw_pred[component]
        final_pred[component] = 2 if (values == 2).sum() > (values == 1).sum() else 1
    return final_pred


@torch.no_grad()
def run_inference(
    damage_model,
    image: torch.Tensor,
    device_name: str,
    damage_tta: str,
    use_building: bool,
    building_model=None,
    building_threshold: float = 0.60,
    building_tta: str = "none",
) -> dict[str, Any]:
    device = torch.device(device_name)
    batch = image.unsqueeze(0).to(device)

    damage_logits = tta_d4_logits(damage_model, batch) if damage_tta == "d4" else damage_model(batch).float()
    probs = torch.softmax(damage_logits, dim=1).squeeze(0).cpu().numpy().transpose(1, 2, 0)
    raw_pred = np.argmax(probs, axis=-1).astype(np.int16)

    building_mask = None
    building_probs = None
    final_pred = raw_pred.copy()

    if use_building:
        if building_model is None:
            raise AppError("Le pipeline building est activé, mais le modèle building n'est pas chargé.")
        building_input = batch[:, :3]
        building_logits = (
            tta_d4_logits(building_model, building_input)
            if building_tta == "d4"
            else building_model(building_input).float()
        )
        building_logits = normalize_building_logits(building_logits)
        building_probs = torch.sigmoid(building_logits).squeeze().cpu().numpy().astype(np.float32)
        building_mask = building_probs >= building_threshold
        final_pred = component_majority(raw_pred, building_mask)

    return {
        "raw_pred": raw_pred,
        "final_pred": final_pred.astype(np.int16),
        "probs": probs,
        "building_mask": building_mask,
        "building_probs": building_probs,
    }


def build_result_record(
    *,
    pre: np.ndarray,
    post: np.ndarray,
    inference: dict[str, Any],
    target: np.ndarray | None,
    pair_id: str,
    model_name: str,
    pipeline_label: str,
) -> dict[str, Any]:
    return {
        "pre": pre,
        "post": post,
        "raw_pred": inference["raw_pred"],
        "final_pred": inference["final_pred"],
        "probs": inference["probs"],
        "building_mask": inference["building_mask"],
        "building_probs": inference["building_probs"],
        "target": target,
        "pair_id": pair_id,
        "model_name": model_name,
        "pipeline_label": pipeline_label,
    }


# == Interface =================================================================


def render_header(theme: str, model_label: str, device: str, pipeline_label: str) -> None:
    theme_badge = {"Sombre": "Dark", "Clair": "Light", "Système": "System"}.get(theme, theme)
    st.markdown(
        f"""
<div class="amr-header">
  <div>
    <div class="amr-logo">AFTERMATH</div>
    <div class="amr-tagline">Voir les dégâts pour agir plus vite</div>
  </div>
  <div class="amr-badges">
    <span class="amr-badge accent">{model_label}</span>
    <span class="amr-badge">{pipeline_label}</span>
    <span class="amr-badge">{device.upper()}</span>
    <span class="amr-badge">{theme_badge}</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_section(label: str) -> None:
    st.markdown(f'<div class="section-lbl">{label}</div>', unsafe_allow_html=True)


def render_legend() -> None:
    st.markdown(
        """
<div class="legend-bar">
  <div class="leg-item"><span class="leg-dot" style="background:#000;border:1px solid #555;"></span>Fond</div>
  <div class="leg-item"><span class="leg-dot" style="background:#00aa50;"></span>Bâtiment intact</div>
  <div class="leg-item"><span class="leg-dot" style="background:#dc2828;"></span>Bâtiment endommagé</div>
  <div class="leg-item"><span class="leg-dot" style="background:#00b4dc;"></span>Masque bâtiment</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_disaster_tag(pair_id: str) -> None:
    info = get_disaster(pair_id)
    if not info:
        return
    disaster, place, color = info
    label = disaster.replace("-", " ").title()
    st.markdown(
        f'<span style="display:inline-block;border:1px solid {color};color:{color};'
        f'border-radius:999px;padding:0.18rem 0.65rem;font-size:0.76rem;">'
        f'{label} · {place}</span>',
        unsafe_allow_html=True,
    )


def mcard(value: str, label: str) -> str:
    return f'<div class="mcard"><div class="mval">{value}</div><div class="mlbl">{label}</div></div>'


def render_metrics_grid(metrics: dict[str, Any], with_iou: bool) -> None:
    cards = [
        mcard(f"{metrics.get('damage_ratio', 0.0):.1%}", "Taux dommage"),
        mcard(f"{int(metrics.get('building_pixels', 0)):,}", "Pixels bâtiment"),
        mcard(f"{int(metrics.get('damaged_pixels', 0)):,}", "Pixels endommagés"),
    ]
    if with_iou:
        cards.extend(
            [
                mcard(f"{metrics.get('pixel_accuracy', 0.0):.3f}", "Pixel accuracy"),
                mcard(f"{metrics.get('mean_iou', 0.0):.3f}", "Mean IoU"),
                mcard(f"{metrics.get('iou_damaged', 0.0):.3f}", "IoU damaged"),
                mcard(f"{metrics.get('f1_damaged', 0.0):.3f}", "F1 damaged"),
            ]
        )
    st.markdown(f'<div class="metric-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def render_class_chart(pred: np.ndarray) -> None:
    total = pred.size
    labels = ["Fond", "Intact", "Endommagé"]
    counts = [int((pred == class_id).sum()) for class_id in range(3)]
    if not PLOTLY_AVAILABLE:
        for label, count in zip(labels, counts):
            st.write(f"**{label}** : {count:,} px ({count / total:.1%})")
        return
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=[count / total * 100 for count in counts],
            marker_color=["#1e293b", "#00aa50", "#dc2828"],
            text=[f"{count / total:.1%}" for count in counts],
            textposition="outside",
        )
    )
    fig.update_layout(
        height=230,
        margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(showgrid=False, showticklabels=False),
        xaxis=dict(showgrid=False),
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch")


def render_comparison_slider(
    left_image: np.ndarray,
    right_image: np.ndarray,
    left_label: str = "Après",
    right_label: str = "Overlay final",
) -> None:
    height, width = left_image.shape[:2]
    container_height = min(int(900 * height / width), 620)
    b64_left = img_to_b64(left_image)
    b64_right = img_to_b64(right_image)
    html = f"""
<div id="cmp" style="position:relative;width:100%;height:{container_height}px;overflow:hidden;border-radius:10px;cursor:col-resize;background:#000;">
  <img src="data:image/png;base64,{b64_left}" style="position:absolute;inset:0;width:100%;height:100%;object-fit:contain;">
  <div id="cmp-over" style="position:absolute;inset:0;width:50%;overflow:hidden;">
    <img id="cmp-right" src="data:image/png;base64,{b64_right}" style="position:absolute;top:0;left:0;height:100%;object-fit:contain;">
  </div>
  <div id="cmp-bar" style="position:absolute;top:0;left:50%;width:2px;height:100%;background:white;box-shadow:0 0 8px rgba(0,0,0,.45);"></div>
  <div style="position:absolute;bottom:10px;left:12px;background:rgba(0,0,0,.65);color:white;font-size:12px;padding:4px 10px;border-radius:6px;">{left_label}</div>
  <div style="position:absolute;bottom:10px;right:12px;background:rgba(220,40,40,.85);color:white;font-size:12px;padding:4px 10px;border-radius:6px;">{right_label}</div>
</div>
<script>
(function() {{
  const cmp = document.getElementById('cmp');
  const over = document.getElementById('cmp-over');
  const bar = document.getElementById('cmp-bar');
  const right = document.getElementById('cmp-right');
  let dragging = false;
  function syncWidth() {{ right.style.width = cmp.getBoundingClientRect().width + 'px'; }}
  function setPos(evt) {{
    const rect = cmp.getBoundingClientRect();
    const clientX = evt.touches ? evt.touches[0].clientX : evt.clientX;
    const x = Math.max(0, Math.min(clientX - rect.left, rect.width));
    const pct = x / rect.width * 100;
    over.style.width = pct + '%';
    bar.style.left = pct + '%';
    syncWidth();
  }}
  cmp.addEventListener('mousedown', e => {{ dragging = true; setPos(e); }});
  document.addEventListener('mouseup', () => dragging = false);
  document.addEventListener('mousemove', e => {{ if (dragging) setPos(e); }});
  cmp.addEventListener('touchstart', e => {{ dragging = true; setPos(e); }}, {{passive:true}});
  document.addEventListener('touchend', () => dragging = false);
  document.addEventListener('touchmove', e => {{ if (dragging) setPos(e); }}, {{passive:true}});
  window.addEventListener('load', syncWidth);
  setTimeout(syncWidth, 200);
}})();
</script>
"""
    components.html(html, height=container_height + 12)


def render_download(
    final_pred: np.ndarray,
    post: np.ndarray,
    metrics: dict[str, Any] | None,
    pair_id: str,
    model_name: str,
) -> None:
    col_mask, col_overlay, col_json = st.columns(3)

    mask_buffer = io.BytesIO()
    Image.fromarray(colorize(final_pred)).save(mask_buffer, format="PNG")
    col_mask.download_button(
        "Masque PNG",
        mask_buffer.getvalue(),
        file_name=f"mask_{pair_id}_{datetime.now():%H%M%S}.png",
        mime="image/png",
        width="stretch",
    )

    overlay_buffer = io.BytesIO()
    Image.fromarray(make_overlay(post, final_pred)).save(overlay_buffer, format="PNG")
    col_overlay.download_button(
        "Overlay PNG",
        overlay_buffer.getvalue(),
        file_name=f"overlay_{pair_id}_{datetime.now():%H%M%S}.png",
        mime="image/png",
        width="stretch",
    )

    report = {
        "generated_at": datetime.now().isoformat(),
        "pair_id": pair_id,
        "model": model_name,
        "metrics": metrics or {},
        "target_mode": TARGET_MODE,
    }
    col_json.download_button(
        "Rapport JSON",
        json.dumps(report, indent=2, ensure_ascii=False),
        file_name=f"report_{pair_id}_{datetime.now():%H%M%S}.json",
        mime="application/json",
        width="stretch",
    )


def add_to_history(pair_id: str, model_name: str, pipeline_label: str, stats: dict[str, Any]) -> None:
    history = st.session_state.setdefault("history", [])
    history.append(
        {
            "pair_id": pair_id,
            "model": model_name,
            "pipeline": pipeline_label,
            "time": datetime.now().strftime("%H:%M:%S"),
            "damage_ratio": float(stats.get("damage_ratio", 0.0)),
        }
    )


def render_history() -> None:
    history = st.session_state.get("history", [])
    if not history:
        st.caption("Aucune analyse dans cette session.")
        return
    for item in reversed(history[-8:]):
        st.markdown(
            f"""
<div class="history-row">
  <div>
    <div class="hr-id">{item['pair_id']}</div>
    <div class="hr-meta">{item['model']} · {item['pipeline']} · {item['time']}</div>
  </div>
  <div style="text-align:right;">
    <div class="hr-dmg">{item['damage_ratio']:.1%}</div>
    <div class="hr-meta">dommage</div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )


def render_results(record: dict[str, Any], use_building: bool) -> None:
    pre = record["pre"]
    post = record["post"]
    raw_pred = record["raw_pred"]
    final_pred = record["final_pred"]
    probs = record["probs"]
    building_mask = record["building_mask"]
    building_probs = record["building_probs"]
    target = record["target"]
    pair_id = record["pair_id"]
    model_name = record["model_name"]
    pipeline_label = record["pipeline_label"]

    metrics = compute_metrics(final_pred, target) if target is not None else None
    stats = metrics or prediction_stats(final_pred)
    overlay_final = make_overlay(post, final_pred)

    render_legend()
    tabs = st.tabs(["Visualisation", "Métriques", "Incertitude du modèle", "Exporter"])

    with tabs[0]:
        render_disaster_tag(pair_id)
        st.caption(f"Pipeline sélectionné : {pipeline_label}")
        if use_building:
            st.warning(
                "Le post-processing bâtiment peut améliorer la précision, mais il peut aussi retirer "
                "de vrais pixels endommagés si le masque bâtiment manque une structure."
            )

        render_section("Résultat principal")
        render_comparison_slider(post, overlay_final, "Après catastrophe", "Overlay final")

        render_section("Explicabilité visuelle")
        row1 = st.columns(3)
        row1[0].image(pre, caption="Image avant catastrophe", width="stretch")
        row1[1].image(post, caption="Image après catastrophe", width="stretch")
        row1[2].image(colorize(raw_pred), caption="Damage brut / TTA d4", width="stretch")

        row2 = st.columns(3)
        if building_mask is not None:
            row2[0].image(colorize_building(building_mask), caption="Masque bâtiment prédit", width="stretch")
        else:
            row2[0].image(np.zeros_like(post), caption="Masque bâtiment non utilisé", width="stretch")
        row2[1].image(colorize(final_pred), caption="Damage final post-processé", width="stretch")
        if target is not None:
            row2[2].image(colorize(target), caption="Vérité terrain", width="stretch")
        else:
            row2[2].image(overlay_final, caption="Overlay final", width="stretch")

    with tabs[1]:
        if metrics is not None:
            render_section("Métriques de la paire")
            render_metrics_grid(metrics, with_iou=True)
            left, right = st.columns(2)
            with left:
                render_section("Distribution finale prédite")
                render_class_chart(final_pred)
            with right:
                render_section("Distribution vérité terrain")
                render_class_chart(target)
        else:
            render_section("Statistiques de prédiction")
            render_metrics_grid(stats, with_iou=False)
            st.info("Aucune vérité terrain fournie - inférence uniquement.")
            render_section("Distribution finale prédite")
            render_class_chart(final_pred)

        if building_probs is not None:
            render_section("Probabilité bâtiment")
            st.caption(f"Pixels bâtiment retenus : {int(building_mask.sum()) if building_mask is not None else 0:,}")
            st.image((np.clip(building_probs, 0, 1) * 255).astype(np.uint8), width="stretch")

    with tabs[2]:
        render_section("Carte d'entropie")
        st.markdown(
            """
<div class="info-strip">
L'entropie visualise l'incertitude du modèle pixel par pixel. Les zones chaudes sont
souvent des contours de bâtiments, des textures ambiguës ou des régions où le modèle
hésite entre bâtiment intact et bâtiment endommagé.
</div>
""",
            unsafe_allow_html=True,
        )
        st.image(entropy_map(probs), caption="Entropie normalisée", width="stretch")

        render_section("Probabilités par classe")
        prob_cols = st.columns(3)
        for class_id, label in enumerate(["Fond", "Bâtiment intact", "Bâtiment endommagé"]):
            prob_cols[class_id].image(
                (probs[:, :, class_id] * 255).astype(np.uint8),
                caption=label,
                width="stretch",
            )

    with tabs[3]:
        render_section("Télécharger les résultats")
        render_download(final_pred, post, metrics, pair_id, model_name)
        render_section("Historique")
        render_history()


def render_sidebar() -> dict[str, Any]:
    st.sidebar.markdown("**Thème**")
    theme = st.sidebar.radio(
        "Thème",
        THEME_LABELS,
        horizontal=True,
        label_visibility="collapsed",
        key="theme_choice",
    )
    st.sidebar.divider()

    st.sidebar.markdown("**Modèle damage**")
    model_id = st.sidebar.selectbox(
        "Modèle damage",
        list(DAMAGE_MODELS),
        index=0,
        format_func=lambda key: DAMAGE_MODELS[key]["label"],
        label_visibility="collapsed",
    )
    model_cfg = DAMAGE_MODELS[model_id]
    st.sidebar.caption(model_cfg["description"])
    if model_cfg["checkpoint"].exists():
        st.sidebar.success("Checkpoint damage disponible.")
    else:
        st.sidebar.error(f"Checkpoint damage manquant : {rel_path(model_cfg['checkpoint'])}")
    st.sidebar.caption(f"Checkpoint utilisé : {rel_path(model_cfg['checkpoint'])}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cuda_available = device == "cuda"
    building_available = BUILDING_MODULE_AVAILABLE and BUILDING_CHECKPOINT.exists()

    st.sidebar.divider()
    st.sidebar.markdown("**Résumé modèle**")
    st.sidebar.caption("Damage champion v2 : F1 damaged = 0.7013")
    st.sidebar.caption("Building b400 : F1 building = 0.8504")
    if cuda_available and building_available:
        st.sidebar.success("Pipeline recommandé : Qualité maximale")
    else:
        st.sidebar.info("Pipeline recommandé : Qualité si CUDA ou building indisponible")

    st.sidebar.divider()
    st.sidebar.markdown("**Modèle bâtiment**")
    st.sidebar.caption(f"Architecture : {BUILDING_MODEL_NAME}")
    if building_available:
        st.sidebar.success("Checkpoint building b400 disponible.")
        st.sidebar.caption(rel_path(BUILDING_CHECKPOINT))
    else:
        st.sidebar.warning("Checkpoint building b400 non disponible.")
        st.sidebar.caption(rel_path(BUILDING_CHECKPOINT))

    st.sidebar.divider()
    st.sidebar.markdown("**Pipeline**")
    pipeline_ids = list(PIPELINES)
    default_pipeline_index = pipeline_ids.index("damage_tta_building") if cuda_available and building_available else pipeline_ids.index("damage_tta")
    pipeline_id = st.sidebar.selectbox(
        "Pipeline",
        pipeline_ids,
        index=default_pipeline_index,
        format_func=lambda key: PIPELINES[key]["label"],
        label_visibility="collapsed",
    )
    pipeline_cfg = PIPELINES[pipeline_id]
    st.sidebar.caption(pipeline_cfg["description"])

    building_threshold = 0.60
    building_tta = "none"
    pipeline_ready = True
    if pipeline_cfg["use_building"]:
        if not building_available:
            st.sidebar.error("Le pipeline qualité maximale exige le checkpoint building.")
            pipeline_ready = False
        building_threshold = st.sidebar.slider("Seuil bâtiment", 0.10, 0.90, 0.60, 0.05)
        use_building_tta = st.sidebar.toggle("TTA d4 building", value=cuda_available)
        building_tta = "d4" if use_building_tta else "none"

    if not cuda_available:
        st.sidebar.warning("CUDA indisponible : l'inférence CPU sera lente.")

    st.sidebar.divider()
    st.sidebar.markdown("**Source de données**")
    source_keys = list(SOURCE_MODES)
    default_source_index = source_keys.index("embedded") if list_embedded_pairs() else source_keys.index("upload")
    source_mode = st.sidebar.radio(
        "Source",
        source_keys,
        index=default_source_index,
        format_func=lambda key: SOURCE_MODES[key],
        label_visibility="collapsed",
    )

    st.sidebar.divider()
    st.sidebar.caption(f"Appareil : {device.upper()} · Image : {model_cfg['image_size']} px · Target : {TARGET_MODE}")

    return {
        "theme": theme,
        "device": device,
        "model_id": model_id,
        "model_cfg": model_cfg,
        "model_label": model_cfg["label"],
        "pipeline_id": pipeline_id,
        "pipeline_label": pipeline_cfg["label"],
        "pipeline_ready": pipeline_ready,
        "damage_tta": pipeline_cfg["damage_tta"],
        "use_building": bool(pipeline_cfg["use_building"]),
        "building_threshold": building_threshold,
        "building_tta": building_tta,
        "source_mode": source_mode,
    }


def prepare_models(cfg: dict[str, Any]):
    model_cfg = cfg["model_cfg"]
    damage_model = load_damage_model(
        str(model_cfg["checkpoint"]),
        model_cfg["model_name"],
        int(model_cfg["in_channels"]),
        int(model_cfg["base_channels"]),
        cfg["device"],
    )
    building_model = load_building_model(cfg["device"]) if cfg["use_building"] else None
    return damage_model, building_model


def list_embedded_pairs() -> list[str]:
    if not SAMPLE_PAIRS_ROOT.exists():
        return []
    return sorted(
        path.name
        for path in SAMPLE_PAIRS_ROOT.iterdir()
        if path.is_dir() and (path / "pre.png").exists() and (path / "post.png").exists()
    )


def render_embedded_mode(cfg: dict[str, Any]) -> None:
    render_section("Exemples inclus dans le dépôt")
    pair_ids = list_embedded_pairs()
    if not pair_ids:
        st.error(
            "Aucun exemple embarqué trouvé. Vérifiez le dossier "
            f"`{SAMPLE_PAIRS_ROOT.relative_to(PROJECT_ROOT)}` ou utilisez le mode upload."
        )
        return

    st.info(
        "Ces paires légères permettent de tester Aftermath sans télécharger le dataset xBD complet. "
        "Elles servent uniquement à la démonstration."
    )

    col_select, col_button = st.columns([4, 1])
    pair_id = col_select.selectbox("Exemple inclus", pair_ids, label_visibility="collapsed")
    pair_dir = SAMPLE_PAIRS_ROOT / pair_id
    analyze = col_button.button("Analyser", type="primary", width="stretch")

    size = int(cfg["model_cfg"]["image_size"])
    try:
        pre_np = read_rgb_path(pair_dir / "pre.png", size)
        post_np = read_rgb_path(pair_dir / "post.png", size)
    except OSError as exc:
        st.error(f"Impossible de lire l'exemple embarqué `{pair_id}` : {exc}")
        return

    preview_cols = st.columns(2)
    preview_cols[0].image(pre_np, caption="Avant catastrophe", width="stretch")
    preview_cols[1].image(post_np, caption="Après catastrophe", width="stretch")

    cache_key = (
        "embedded",
        pair_id,
        cfg["model_id"],
        cfg["pipeline_id"],
        cfg["building_threshold"],
        cfg["building_tta"],
    )
    if analyze:
        st.session_state["embedded_pending_key"] = cache_key

    if (
        st.session_state.get("embedded_pending_key") == cache_key
        and st.session_state.get("embedded_done_key") != cache_key
    ):
        with st.spinner("Analyse en cours..."):
            try:
                stacked = np.concatenate([pre_np, post_np], axis=2).transpose(2, 0, 1)
                image = torch.from_numpy(stacked.copy()).float().div(255.0)
                damage_model, building_model = prepare_models(cfg)
                inference = run_inference(
                    damage_model,
                    image,
                    cfg["device"],
                    cfg["damage_tta"],
                    cfg["use_building"],
                    building_model,
                    cfg["building_threshold"],
                    cfg["building_tta"],
                )
            except (AppError, RuntimeError, OSError, ValueError) as exc:
                st.error(str(exc))
                return

        target = read_demo_target(pair_dir / "target.png", size) if (pair_dir / "target.png").exists() else None
        record = build_result_record(
            pre=pre_np,
            post=post_np,
            inference=inference,
            target=target,
            pair_id=pair_id,
            model_name=cfg["model_label"],
            pipeline_label=cfg["pipeline_label"],
        )
        st.session_state["embedded_cache_key"] = cache_key
        st.session_state["embedded_cache"] = record
        st.session_state["embedded_done_key"] = cache_key
        stats = compute_metrics(record["final_pred"], target) if target is not None else prediction_stats(record["final_pred"])
        add_to_history(pair_id, cfg["model_label"], cfg["pipeline_label"], stats)

    record = None
    if st.session_state.get("embedded_cache_key") == cache_key:
        record = st.session_state.get("embedded_cache")
    if record is None:
        st.info("Sélectionnez un exemple inclus et cliquez sur Analyser.")
        return
    render_results(record, cfg["use_building"])


def render_dataset_mode(cfg: dict[str, Any]) -> None:
    if not DATA_ROOT.exists():
        st.error(f"Données xBD introuvables : `{DATA_ROOT}`")
        return

    all_ids: list[str] = []
    for split in ("test", "val", "train"):
        try:
            all_ids.extend(load_split(split)["pair_id"].astype(str).tolist())
        except AppError:
            continue
    if not all_ids:
        st.error("Aucune paire disponible dans les splits.")
        return

    seen: set[str] = set()
    unique_ids = [pair_id for pair_id in all_ids if not (pair_id in seen or seen.add(pair_id))]
    recommended = [pair_id for pair_id in RECOMMENDED_PAIR_IDS if pair_id in set(unique_ids)]
    show_recommended = st.sidebar.toggle("Exemples recommandés uniquement", value=bool(recommended))
    pair_ids = recommended if show_recommended and recommended else unique_ids

    col_select, col_button = st.columns([4, 1])
    pair_id = col_select.selectbox("Paire xBD", pair_ids, label_visibility="collapsed")
    analyze = col_button.button("Analyser", type="primary", width="stretch")

    cache_key = (
        "dataset",
        pair_id,
        cfg["model_id"],
        cfg["pipeline_id"],
        cfg["building_threshold"],
        cfg["building_tta"],
    )
    if analyze:
        st.session_state["dataset_pending_key"] = cache_key

    if st.session_state.get("dataset_pending_key") == cache_key and st.session_state.get("dataset_done_key") != cache_key:
        split_for_pair = "test"
        for split in ("test", "val", "train"):
            try:
                if pair_id in load_split(split)["pair_id"].astype(str).values:
                    split_for_pair = split
                    break
            except AppError:
                continue

        with st.spinner("Analyse en cours..."):
            try:
                sample = load_sample(split_for_pair, pair_id, int(cfg["model_cfg"]["image_size"]))
                damage_model, building_model = prepare_models(cfg)
                inference = run_inference(
                    damage_model,
                    sample["image"],
                    cfg["device"],
                    cfg["damage_tta"],
                    cfg["use_building"],
                    building_model,
                    cfg["building_threshold"],
                    cfg["building_tta"],
                )
            except (AppError, XBDDatasetError, RuntimeError, OSError, ValueError) as exc:
                st.error(str(exc))
                return

        target = sample["target"].detach().cpu().numpy()
        record = build_result_record(
            pre=tensor_to_rgb(sample["image"][:3]),
            post=tensor_to_rgb(sample["image"][3:6]),
            inference=inference,
            target=target,
            pair_id=pair_id,
            model_name=cfg["model_label"],
            pipeline_label=cfg["pipeline_label"],
        )
        st.session_state["dataset_cache_key"] = cache_key
        st.session_state["dataset_cache"] = record
        st.session_state["dataset_done_key"] = cache_key
        add_to_history(pair_id, cfg["model_label"], cfg["pipeline_label"], compute_metrics(record["final_pred"], target))

    record = None
    if st.session_state.get("dataset_cache_key") == cache_key:
        record = st.session_state.get("dataset_cache")
    if record is None:
        st.info("Sélectionnez une paire et cliquez sur Analyser.")
        return
    render_results(record, cfg["use_building"])


def render_upload_mode(cfg: dict[str, Any]) -> None:
    render_section("Téléverser une paire d'images satellite")
    left, right = st.columns(2)

    with left:
        st.markdown("**Image avant catastrophe**")
        pre_file = st.file_uploader(
            "Image avant catastrophe",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            key="upload_pre",
            label_visibility="collapsed",
        )
        if pre_file:
            st.image(read_upload(pre_file, int(cfg["model_cfg"]["image_size"])), caption="Aperçu - Avant", width="stretch")

    with right:
        st.markdown("**Image après catastrophe**")
        post_file = st.file_uploader(
            "Image après catastrophe",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            key="upload_post",
            label_visibility="collapsed",
        )
        if post_file:
            st.image(read_upload(post_file, int(cfg["model_cfg"]["image_size"])), caption="Aperçu - Après", width="stretch")

    both_ready = pre_file is not None and post_file is not None
    if not both_ready:
        st.info("Chargez une image avant et une image après pour activer l'inférence.")

    analyze = st.button(
        "Lancer l'inférence" if both_ready else "En attente des deux images",
        type="primary",
        disabled=not both_ready,
        width="stretch",
    )

    upload_key = None
    if both_ready:
        upload_key = (
            "upload",
            pre_file.name,
            pre_file.size,
            post_file.name,
            post_file.size,
            cfg["model_id"],
            cfg["pipeline_id"],
            cfg["building_threshold"],
            cfg["building_tta"],
        )

    has_upload_cache = (
        upload_key is not None
        and st.session_state.get("upload_cache_key") == upload_key
        and "upload_cache" in st.session_state
    )

    if has_upload_cache and not analyze:
        render_results(st.session_state["upload_cache"], cfg["use_building"])
        return

    if not analyze:
        return

    with st.spinner("Analyse satellite en cours..."):
        try:
            size = int(cfg["model_cfg"]["image_size"])
            pre_np = read_upload(pre_file, size)
            post_np = read_upload(post_file, size)
            stacked = np.concatenate([pre_np, post_np], axis=2).transpose(2, 0, 1)
            image = torch.from_numpy(stacked.copy()).float().div(255.0)
            damage_model, building_model = prepare_models(cfg)
            inference = run_inference(
                damage_model,
                image,
                cfg["device"],
                cfg["damage_tta"],
                cfg["use_building"],
                building_model,
                cfg["building_threshold"],
                cfg["building_tta"],
            )
        except (AppError, RuntimeError, OSError, ValueError) as exc:
            st.error(str(exc))
            return

    pair_id = f"upload_{datetime.now():%Y%m%d_%H%M%S}"
    record = build_result_record(
        pre=pre_np,
        post=post_np,
        inference=inference,
        target=None,
        pair_id=pair_id,
        model_name=cfg["model_label"],
        pipeline_label=cfg["pipeline_label"],
    )
    st.session_state["upload_cache_key"] = upload_key
    st.session_state["upload_cache"] = record
    add_to_history(pair_id, cfg["model_label"], cfg["pipeline_label"], prediction_stats(record["final_pred"]))
    render_results(record, cfg["use_building"])


def main() -> None:
    st.set_page_config(
        page_title="Aftermath",
        page_icon="satellite",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    cfg = render_sidebar()
    st.markdown(build_css(cfg["theme"]), unsafe_allow_html=True)
    render_header(cfg["theme"], cfg["model_label"], cfg["device"], cfg["pipeline_label"])

    if not cfg["model_cfg"]["checkpoint"].exists():
        st.error(f"Checkpoint damage introuvable : `{cfg['model_cfg']['checkpoint']}`")
        st.stop()
    if not cfg["pipeline_ready"]:
        st.error("Pipeline sélectionné indisponible : vérifiez le checkpoint building ou choisissez un autre mode.")
        st.stop()

    if cfg["source_mode"] == "upload":
        render_upload_mode(cfg)
    elif cfg["source_mode"] == "embedded":
        render_embedded_mode(cfg)
    elif cfg["source_mode"] == "dataset":
        render_dataset_mode(cfg)
    else:
        st.error(f"Mode source inconnu : {cfg['source_mode']}")


if __name__ == "__main__":
    main()
