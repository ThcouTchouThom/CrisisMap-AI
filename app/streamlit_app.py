"""Streamlit prototype for CrisisMap AI U-Net inference."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from crisismap.data.xbd_dataset import (  # noqa: E402
    XBDDatasetError,
    XBDPairDataset as XBDDataset,
)
from crisismap.models.unet import UNet  # noqa: E402


DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "xbd" / "train"
SPLITS_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
    / "unet_baseline"
    / "best_unet.pt"
)
SAM_CHECKPOINT_PATH = PROJECT_ROOT / "src" / "crisismap" / "models" / "sam" / "sam_vit_b_01ec64.pth"

IMAGE_SIZE = 512
TARGET_MODE = "3-class"
BASE_CHANNELS = 32

CHECKPOINT_LABEL = "unet_baseline/best_unet.pt"
RECOMMENDED_PAIR_IDS_BY_SPLIT = {
    "train": [
        "hurricane-harvey_00000000",
        "hurricane-michael_00000034",
        "santa-rosa-wildfire_00000035",
        "palu-tsunami_00000042",
    ],
    "val": [
        "hurricane-harvey_00000137",
        "hurricane-harvey_00000347",
        "santa-rosa-wildfire_00000129",
    ],
    "test": [
        "hurricane-harvey_00000358",
        "hurricane-harvey_00000132",
        "santa-rosa-wildfire_00000093",
        "palu-tsunami_00000109",
        "hurricane-michael_00000120",
    ],
}

CLASS_COLORS = {
    0: np.array([0, 0, 0], dtype=np.uint8),
    1: np.array([0, 170, 80], dtype=np.uint8),
    2: np.array([220, 40, 40], dtype=np.uint8),
}


class AppError(Exception):
    """Raised when the Streamlit prototype cannot continue safely."""


@st.cache_data(show_spinner=False)
def load_split(split_name: str) -> pd.DataFrame:
    split_csv = SPLITS_DIR / f"{split_name}_pairs.csv"
    if not split_csv.exists():
        raise AppError(f"Split CSV not found: {split_csv}")
    if not split_csv.is_file():
        raise AppError(f"Split path is not a file: {split_csv}")

    try:
        df = pd.read_csv(split_csv)
    except OSError as exc:
        raise AppError(f"Could not read split CSV '{split_csv}': {exc}") from exc
    except pd.errors.EmptyDataError as exc:
        raise AppError(f"Split CSV is empty: {split_csv}") from exc
    except pd.errors.ParserError as exc:
        raise AppError(f"Could not parse split CSV '{split_csv}': {exc}") from exc

    if "pair_id" not in df.columns:
        raise AppError(f"Split CSV is missing required column: pair_id")
    if df.empty:
        raise AppError(f"Split CSV has no rows: {split_csv}")
    return df


@st.cache_resource(show_spinner="Chargement du checkpoint U-Net...")
def load_model(checkpoint_path: str, device_name: str) -> UNet:
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise AppError(f"Checkpoint not found: {checkpoint}")
    if not checkpoint.is_file():
        raise AppError(f"Checkpoint path is not a file: {checkpoint}")

    device = torch.device(device_name)
    model = UNet(in_channels=6, num_classes=3, base_channels=BASE_CHANNELS).to(device)

    try:
        loaded = load_checkpoint_file(checkpoint, device)
        state_dict = extract_state_dict(loaded)
        model.load_state_dict(state_dict)
    except OSError as exc:
        raise AppError(f"Could not read checkpoint '{checkpoint}': {exc}") from exc
    except RuntimeError as exc:
        raise AppError(
            "Checkpoint weights do not match the expected 3-class U-Net "
            f"configuration at {checkpoint}."
        ) from exc

    model.eval()
    return model


@st.cache_resource(show_spinner="Chargement du modèle SAM")
def load_sam_refiner(sam_checkpoint_path: str, device_name: str):
    """Charge le SAMRefiner. Retourne None si SAM n'est pas disponible."""
    try:
        from crisismap.models.sam_refiner import SAMRefiner  # noqa: E402
    except ImportError:
        st.warning("Le module SAMRefiner est introuvable dans crisismap/models/.")
        return None

    checkpoint = Path(sam_checkpoint_path)
    if not checkpoint.exists():
        st.warning(f"Checkpoint SAM introuvable : `{checkpoint}`")
        return None

    try:
        device = torch.device(device_name)
        return SAMRefiner(
            checkpoint=str(checkpoint),
            model_type="configs/sam2.1/sam2.1_hiera_l.yaml",
            device=device,
            building_classes={1, 2},
        )
    except Exception as exc:
        st.warning(f"Impossible de charger SAM : {exc}")
        return None


def load_checkpoint_file(path: Path, device: torch.device) -> object:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def extract_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise AppError("Checkpoint is not a state_dict or model checkpoint dict.")

    cleaned = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            cleaned[key.removeprefix("module.")] = value
    if not cleaned:
        raise AppError("Checkpoint does not contain tensor weights.")
    return cleaned


def load_sample(split_name: str, pair_id: str) -> dict[str, object]:
    split_csv = SPLITS_DIR / f"{split_name}_pairs.csv"
    dataset = XBDDataset(
        root=DATA_ROOT,
        split_csv=split_csv,
        image_size=IMAGE_SIZE,
        target_mode=TARGET_MODE,
    )
    matches = dataset.samples.index[
        dataset.samples["pair_id"].astype(str) == str(pair_id)
    ].tolist()
    if not matches:
        raise AppError(f"Pair id '{pair_id}' was not found in {split_csv}.")
    return dataset[int(matches[0])]


@torch.no_grad()
def predict(model: UNet, image: torch.Tensor, device_name: str) -> torch.Tensor:
    device = torch.device(device_name)
    logits = model(image.unsqueeze(0).to(device))
    return torch.argmax(logits, dim=1).squeeze(0).cpu()


@torch.no_grad()
def predict_with_sam(
    model: UNet,
    image: torch.Tensor,
    device_name: str,
    sam_refiner,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Retourne (pred_unet, pred_sam) — deux masques (H, W)."""
    device = torch.device(device_name)
    logits = model(image.unsqueeze(0).to(device))
    unet_pred = torch.argmax(logits, dim=1).squeeze(0).cpu()

    # Image post seule (canaux 3-5) pour SAM
    post_image = image[3:].unsqueeze(0)  # (1, 3, H, W)
    sam_pred = sam_refiner.refine_batch(
        post_images=post_image,
        unet_preds=unet_pred.unsqueeze(0),
    ).squeeze(0)

    return unet_pred, sam_pred


def tensor_to_rgb(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().numpy()
    array = np.transpose(array, (1, 2, 0))
    array = np.clip(array, 0.0, 1.0)
    return (array * 255).astype(np.uint8)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    color_image = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        color_image[mask == class_id] = color
    return color_image


def overlay_prediction(post_image: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    color_mask = colorize_mask(prediction).astype(np.float32)
    image = post_image.astype(np.float32)

    alpha = np.zeros(prediction.shape, dtype=np.float32)
    alpha[prediction == 1] = 0.35
    alpha[prediction == 2] = 0.55
    alpha = alpha[:, :, None]

    overlay = image * (1.0 - alpha) + color_mask * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


def prediction_metrics(prediction: np.ndarray) -> tuple[int, int, float]:
    building_pixels = int(np.isin(prediction, [1, 2]).sum())
    damaged_pixels = int((prediction == 2).sum())
    damaged_ratio = damaged_pixels / building_pixels if building_pixels else 0.0
    return building_pixels, damaged_pixels, damaged_ratio


def render_legend() -> None:
    st.markdown(
        """
        <div style="display:flex; gap:1rem; flex-wrap:wrap; margin:0.25rem 0 1rem;">
          <span><span style="background:#000000;display:inline-block;width:14px;height:14px;margin-right:6px;border:1px solid #777;"></span>Fond / absence de bâtiment</span>
          <span><span style="background:#00aa50;display:inline-block;width:14px;height:14px;margin-right:6px;border:1px solid #777;"></span>Bâtiment non endommagé</span>
          <span><span style="background:#dc2828;display:inline-block;width:14px;height:14px;margin-right:6px;border:1px solid #777;"></span>Bâtiment endommagé</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_summary() -> None:
    st.sidebar.markdown("### Modèle actuel")
    st.sidebar.write(f"Checkpoint: `{CHECKPOINT_LABEL}`")
    st.sidebar.write("Pixel accuracy: `0.9189`")
    st.sidebar.write("Mean IoU: `0.6363`")
    st.sidebar.write("IoU damaged: `0.4159`")
    st.sidebar.write("F1 damaged: `0.5875`")


def render_sam_sidebar() -> bool:
    """Affiche les options SAM dans la sidebar. Retourne True si SAM activé."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Raffinement SAM")

    sam_available = SAM_CHECKPOINT_PATH.exists()
    if not sam_available:
        st.sidebar.warning(
            f"Checkpoint SAM introuvable :\n`{SAM_CHECKPOINT_PATH}`\n\n"
            "Téléchargez-le avec :\n"
            "```\nwget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
            " -O models/sam/sam_vit_b_01ec64.pth\n```"
        )
        return False

    use_sam = st.sidebar.toggle(
        "Activer SAM",
        value=False,
        help=(
            "Raffine les contours des bâtiments détectés par le U-Net "
            "à l'aide de Segment Anything Model (SAM ViT-B)."
        ),
    )

    if use_sam:
        st.sidebar.success("SAM activé — les masques seront raffinés.")
        st.sidebar.caption(
            "SAM utilise l'image post-catastrophe pour affiner "
            "les contours de chaque bâtiment détecté par le U-Net."
        )
    else:
        st.sidebar.info("SAM désactivé — prédiction U-Net seule.")

    return use_sam


def validate_required_paths() -> None:
    if not DATA_ROOT.exists():
        raise AppError(f"xBD training folder not found: {DATA_ROOT}")
    if not DATA_ROOT.is_dir():
        raise AppError(f"xBD training path is not a directory: {DATA_ROOT}")
    if not SPLITS_DIR.exists():
        raise AppError(f"Split directory not found: {SPLITS_DIR}")


def render_metrics_comparison(
    pred_unet: np.ndarray,
    pred_sam: np.ndarray | None,
    target: np.ndarray,
) -> None:
    bp_u, dp_u, dr_u = prediction_metrics(pred_unet)
    metrics_unet = compute_iou_metrics(pred_unet, target)

    if pred_sam is not None:
        bp_s, dp_s, dr_s = prediction_metrics(pred_sam)
        metrics_sam = compute_iou_metrics(pred_sam, target)

        st.markdown("#### Métriques de prédiction")

        # Ligne header
        col_label, col_unet, col_sam = st.columns([1.2, 1, 1])
        col_label.markdown("&nbsp;")
        col_unet.markdown("**U-Net seul**")
        col_sam.markdown("**U-Net + SAM2**")

        rows = [
            ("Pixel Accuracy",    f"{metrics_unet['pixel_accuracy']:.4f}",  f"{metrics_sam['pixel_accuracy']:.4f}",  metrics_sam['pixel_accuracy']  - metrics_unet['pixel_accuracy']),
            ("Mean IoU",          f"{metrics_unet['mean_iou']:.4f}",         f"{metrics_sam['mean_iou']:.4f}",         metrics_sam['mean_iou']         - metrics_unet['mean_iou']),
            ("IoU Background",    f"{metrics_unet['iou_background']:.4f}",   f"{metrics_sam['iou_background']:.4f}",   metrics_sam['iou_background']   - metrics_unet['iou_background']),
            ("IoU No-Damage",     f"{metrics_unet['iou_no_damage']:.4f}",    f"{metrics_sam['iou_no_damage']:.4f}",    metrics_sam['iou_no_damage']    - metrics_unet['iou_no_damage']),
            ("IoU Damaged",       f"{metrics_unet['iou_damaged']:.4f}",      f"{metrics_sam['iou_damaged']:.4f}",      metrics_sam['iou_damaged']      - metrics_unet['iou_damaged']),
            ("Pixels bâtiments",  f"{bp_u:,}",                               f"{bp_s:,}",                               bp_s - bp_u),
            ("Pixels endommagés", f"{dp_u:,}",                               f"{dp_s:,}",                               dp_s - dp_u),
            ("Part endommagée",   f"{dr_u:.2%}",                             f"{dr_s:.2%}",                             (dr_s - dr_u) * 100),
        ]

        for label, val_u, val_s, delta in rows:
            c1, c2, c3 = st.columns([1.2, 1, 1])
            c1.markdown(f"**{label}**")
            c2.markdown(val_u)
            arrow = "🟢 +" if delta > 0.001 else ("🔴 " if delta < -0.001 else "⚪ ")
            if isinstance(delta, float):
                c3.markdown(f"{val_s} {arrow}{abs(delta):.4f}")
            else:
                c3.markdown(f"{val_s} {arrow}{abs(delta):,}")
    else:
        metric_cols = st.columns(3)
        metric_cols[0].metric("Pixels bâtiments prédits", f"{bp_u:,}")
        metric_cols[1].metric("Pixels endommagés prédits", f"{dp_u:,}")
        metric_cols[2].metric("Part endommagée prédite", f"{dr_u:.2%}")

        metrics_unet = compute_iou_metrics(pred_unet, target)
        m_cols = st.columns(4)
        m_cols[0].metric("Pixel Accuracy", f"{metrics_unet['pixel_accuracy']:.4f}")
        m_cols[1].metric("Mean IoU", f"{metrics_unet['mean_iou']:.4f}")
        m_cols[2].metric("IoU No-Damage", f"{metrics_unet['iou_no_damage']:.4f}")
        m_cols[3].metric("IoU Damaged", f"{metrics_unet['iou_damaged']:.4f}")

def compute_iou_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int = 3,
) -> dict[str, float]:
    """Calcule pixel accuracy, mean IoU et IoU par classe."""
    confusion = np.zeros((num_classes, num_classes), dtype=np.float64)
    for t, p in zip(target.flatten(), pred.flatten()):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            confusion[t, p] += 1

    correct = np.trace(confusion)
    total = confusion.sum()
    pixel_acc = float(correct / total) if total > 0 else 0.0

    intersections = np.diag(confusion)
    unions = confusion.sum(axis=1) + confusion.sum(axis=0) - intersections
    iou_per_class = np.divide(
        intersections,
        unions,
        out=np.full(num_classes, np.nan),
        where=unions > 0,
    )
    mean_iou = float(np.nanmean(iou_per_class))

    return {
        "pixel_accuracy": pixel_acc,
        "mean_iou": mean_iou,
        "iou_background": float(iou_per_class[0]) if not np.isnan(iou_per_class[0]) else 0.0,
        "iou_no_damage": float(iou_per_class[1]) if not np.isnan(iou_per_class[1]) else 0.0,
        "iou_damaged": float(iou_per_class[2]) if not np.isnan(iou_per_class[2]) else 0.0,
    }


def main() -> None:
    st.set_page_config(page_title="Aftermath", layout="wide")
    st.title("Aftermath")
    st.caption("Prototype de cartographie automatique des dommages par imagerie satellite")

    try:
        validate_required_paths()
    except AppError as exc:
        st.error(str(exc))
        st.stop()

    split_name = st.sidebar.selectbox("Jeu de données", ["train", "val", "test"], index=2)
    render_model_summary()

    # --- Options SAM ---
    use_sam = render_sam_sidebar()

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    st.sidebar.markdown("---")
    st.sidebar.write(f"Appareil: `{device_name}`")
    st.sidebar.write(f"Taille d'image: `{IMAGE_SIZE}`")
    st.sidebar.write(f"Mode cible: `{TARGET_MODE}`")

    try:
        split_df = load_split(split_name)
    except AppError as exc:
        st.error(str(exc))
        st.stop()

    all_pair_ids = split_df["pair_id"].astype(str).tolist()
    recommended_for_split = RECOMMENDED_PAIR_IDS_BY_SPLIT.get(split_name, [])
    all_pair_ids_set = set(all_pair_ids)
    available_recommended_ids = [
        pair_id for pair_id in recommended_for_split if pair_id in all_pair_ids_set
    ]
    pair_source = st.sidebar.radio(
        "Sélection des paires",
        ["Toutes les paires disponibles", "Exemples recommandés"],
    )
    if pair_source == "Exemples recommandés" and available_recommended_ids:
        pair_ids = available_recommended_ids
    else:
        pair_ids = all_pair_ids
        if pair_source == "Exemples recommandés":
            st.sidebar.info(
                "Aucun exemple recommandé n'est disponible dans ce jeu de données. "
                "Toutes les paires sont affichées."
            )

    pair_id = st.selectbox("Paire d'images", pair_ids)

    # --- Chargement modèle + inférence ---
    try:
        sample = load_sample(split_name, pair_id)
        model = load_model(str(CHECKPOINT_PATH), device_name)

        if use_sam:
            with st.spinner("Raffinement SAM en cours..."):
                sam_refiner = load_sam_refiner(str(SAM_CHECKPOINT_PATH), device_name)
                if sam_refiner is None:
                    st.error("SAM n'a pas pu être chargé. Basculement en mode U-Net seul.")
                    use_sam = False
                    prediction = predict(model, sample["image"], device_name)
                    pred_sam_np = None
                else:
                    unet_pred, sam_pred = predict_with_sam(
                        model, sample["image"], device_name, sam_refiner
                    )
                    prediction = unet_pred
                    pred_sam_np = sam_pred.numpy()
        else:
            prediction = predict(model, sample["image"], device_name)
            pred_sam_np = None

    except (AppError, XBDDatasetError, RuntimeError, OSError, ValueError) as exc:
        st.error(str(exc))
        st.info(f"Expected checkpoint: `{CHECKPOINT_PATH}`")
        st.stop()

    image_tensor = sample["image"]
    target = sample["target"].detach().cpu().numpy()
    pred_unet_np = prediction.numpy()

    pre_image = tensor_to_rgb(image_tensor[:3])
    post_image = tensor_to_rgb(image_tensor[3:])
    target_mask = colorize_mask(target)
    predicted_mask_unet = colorize_mask(pred_unet_np)
    overlay_unet = overlay_prediction(post_image, pred_unet_np)

    st.subheader(f"Paire sélectionnée : `{pair_id}`")
    render_legend()

    # --- Métriques ---
    render_metrics_comparison(pred_unet_np, pred_sam_np, target)

    # --- Images ---
    st.markdown("#### Images source")
    top_cols = st.columns(2)
    top_cols[0].image(pre_image, caption="Image avant catastrophe", use_container_width=True)
    top_cols[1].image(post_image, caption="Image après catastrophe", use_container_width=True)

    st.markdown("#### Prédictions")

    if use_sam and pred_sam_np is not None:
        predicted_mask_sam = colorize_mask(pred_sam_np)
        overlay_sam = overlay_prediction(post_image, pred_sam_np)

        # Affichage côte à côte U-Net vs SAM
        bottom_cols = st.columns(3)
        bottom_cols[0].image(target_mask, caption="Vérité terrain", use_container_width=True)
        bottom_cols[1].image(
            predicted_mask_unet, caption="U-Net seul", use_container_width=True
        )
        bottom_cols[2].image(
            predicted_mask_sam, caption="U-Net + SAM", use_container_width=True
        )

        st.markdown("#### Superpositions")
        overlay_cols = st.columns(2)
        overlay_cols[0].image(
            overlay_unet,
            caption="Superposition U-Net",
            use_container_width=True,
        )
        overlay_cols[1].image(
            overlay_sam,
            caption="Superposition U-Net + SAM",
            use_container_width=True,
        )
    else:
        bottom_cols = st.columns(3)
        bottom_cols[0].image(target_mask, caption="Vérité terrain", use_container_width=True)
        bottom_cols[1].image(
            predicted_mask_unet,
            caption="Prédiction du modèle",
            use_container_width=True,
        )
        bottom_cols[2].image(
            overlay_unet,
            caption="Superposition sur l'image après catastrophe",
            use_container_width=True,
        )

    with st.expander("Détails de l'échantillon"):
        st.write(f"Lignes du jeu sélectionné: `{len(split_df):,}`")
        st.write(f"Forme du tenseur d'entrée: `{tuple(image_tensor.shape)}`")
        st.write(f"Classes de vérité terrain: `{np.unique(target).tolist()}`")
        st.write(f"Classes prédites (U-Net): `{np.unique(pred_unet_np).tolist()}`")
        if pred_sam_np is not None:
            st.write(f"Classes prédites (SAM): `{np.unique(pred_sam_np).tolist()}`")
        st.write(f"SAM activé: `{use_sam}`")
        st.write(f"Checkpoint SAM: `{SAM_CHECKPOINT_PATH}`")


if __name__ == "__main__":
    main()