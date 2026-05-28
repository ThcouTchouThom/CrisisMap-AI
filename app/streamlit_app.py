"""Streamlit alpha prototype for Aftermath damage-map inference."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image, ImageOps


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
    / "unet_1024_ce_dice_w005_1_4_noleak_match_hist1000_bs2_250epochs"
    / "best_unet_portable.pt"
)

IMAGE_SIZE = 1024
TARGET_MODE = "3-class"
BASE_CHANNELS = 32
CHECKPOINT_LABEL = (
    "unet_1024_ce_dice_w005_1_4_noleak_match_hist1000_bs2_250epochs/"
    "best_unet_portable.pt"
)
TTA_MODE_OPTIONS = {
    "Rapide: none": "none",
    "Équilibré: rot90": "rot90",
    "Qualité maximale: d4": "d4",
}
TTA_MODE_LABELS = {value: label for label, value in TTA_MODE_OPTIONS.items()}

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
JALON3_DEMO_PAIR_IDS = [
    "hurricane-florence_00000070",
    "hurricane-florence_00000217",
    "hurricane-florence_00000153",
]

CLASS_LABELS = {
    0: "Fond / absence de bâtiment",
    1: "Bâtiment non endommagé",
    2: "Bâtiment endommagé",
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
        raise AppError(f"Split CSV introuvable : {split_csv}")
    if not split_csv.is_file():
        raise AppError(f"Le chemin du split n'est pas un fichier : {split_csv}")

    try:
        df = pd.read_csv(split_csv)
    except OSError as exc:
        raise AppError(f"Impossible de lire le split '{split_csv}' : {exc}") from exc
    except pd.errors.EmptyDataError as exc:
        raise AppError(f"Le split est vide : {split_csv}") from exc
    except pd.errors.ParserError as exc:
        raise AppError(f"Impossible de parser le split '{split_csv}' : {exc}") from exc

    if "pair_id" not in df.columns:
        raise AppError("Le split ne contient pas la colonne obligatoire pair_id.")
    if df.empty:
        raise AppError(f"Le split ne contient aucune ligne : {split_csv}")
    return df


@st.cache_resource(show_spinner="Chargement du checkpoint U-Net...")
def load_model(checkpoint_path: str, device_name: str) -> UNet:
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise AppError(f"Checkpoint introuvable : {checkpoint}")
    if not checkpoint.is_file():
        raise AppError(f"Le checkpoint n'est pas un fichier : {checkpoint}")

    device = torch.device(device_name)
    model = UNet(in_channels=6, num_classes=3, base_channels=BASE_CHANNELS).to(device)

    try:
        loaded = load_checkpoint_file(checkpoint, device)
        state_dict = extract_state_dict(loaded)
        model.load_state_dict(state_dict)
    except OSError as exc:
        raise AppError(f"Impossible de lire le checkpoint '{checkpoint}' : {exc}") from exc
    except RuntimeError as exc:
        raise AppError(
            "Les poids du checkpoint ne correspondent pas à la configuration "
            "U-Net 3 classes attendue."
        ) from exc

    model.eval()
    return model


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
        raise AppError("Le checkpoint n'est pas un state_dict valide.")

    cleaned = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            cleaned[key.removeprefix("module.")] = value
    if not cleaned:
        raise AppError("Le checkpoint ne contient aucun tenseur de poids.")
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
        raise AppError(f"La paire '{pair_id}' est absente de {split_csv}.")
    return dataset[int(matches[0])]


@torch.no_grad()
def predict(
    model: UNet,
    image: torch.Tensor,
    device_name: str,
    tta_mode: str,
) -> torch.Tensor:
    device = torch.device(device_name)
    logits = predict_logits_tta(
        model=model,
        image=image.unsqueeze(0).to(device),
        tta_mode=tta_mode,
    )
    return torch.argmax(logits, dim=1).squeeze(0).cpu()


def tta_ops(tta_mode: str) -> list[tuple[int, bool, bool]]:
    if tta_mode == "none":
        return [(0, False, False)]
    if tta_mode == "rot90":
        return [(rotation, False, False) for rotation in range(4)]
    if tta_mode == "d4":
        return [(rotation, False, False) for rotation in range(4)] + [
            (rotation, True, False) for rotation in range(4)
        ]
    raise AppError(f"Mode TTA non pris en charge : {tta_mode}")


def apply_tta_op(tensor: torch.Tensor, op: tuple[int, bool, bool]) -> torch.Tensor:
    rotations, flip_h, flip_v = op
    output = torch.rot90(tensor, k=rotations, dims=(-2, -1)) if rotations else tensor
    if flip_h:
        output = torch.flip(output, dims=(-1,))
    if flip_v:
        output = torch.flip(output, dims=(-2,))
    return output


def invert_tta_op(tensor: torch.Tensor, op: tuple[int, bool, bool]) -> torch.Tensor:
    rotations, flip_h, flip_v = op
    output = tensor
    if flip_v:
        output = torch.flip(output, dims=(-2,))
    if flip_h:
        output = torch.flip(output, dims=(-1,))
    if rotations:
        output = torch.rot90(output, k=-rotations, dims=(-2, -1))
    return output


@torch.no_grad()
def predict_logits_tta(
    model: UNet,
    image: torch.Tensor,
    tta_mode: str,
) -> torch.Tensor:
    logits_sum: torch.Tensor | None = None
    ops = tta_ops(tta_mode)
    for op in ops:
        view = apply_tta_op(image, op)
        logits = model(view)
        logits = invert_tta_op(logits, op).float()
        logits_sum = logits if logits_sum is None else logits_sum + logits
    if logits_sum is None:
        raise AppError(f"Aucune transformation TTA pour le mode : {tta_mode}")
    return logits_sum / float(len(ops))


def uploaded_pair_to_tensor(
    pre_file: object,
    post_file: object,
    image_size: int,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    pre_image = read_uploaded_image(pre_file, image_size)
    post_image = read_uploaded_image(post_file, image_size)
    image = np.concatenate([pre_image, post_image], axis=2)
    tensor = torch.from_numpy(image.transpose(2, 0, 1).copy())
    tensor = tensor.to(dtype=torch.float32).div(255.0)
    return tensor, pre_image, post_image


def read_uploaded_image(uploaded_file: object, image_size: int) -> np.ndarray:
    try:
        with Image.open(uploaded_file) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image = image.resize((image_size, image_size), Image.BILINEAR)
            return np.asarray(image, dtype=np.uint8)
    except OSError as exc:
        raise AppError(f"Impossible de lire l'image téléversée : {exc}") from exc


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


def prediction_counts(prediction: np.ndarray) -> tuple[int, int, float]:
    building_pixels = int(np.isin(prediction, [1, 2]).sum())
    damaged_pixels = int((prediction == 2).sum())
    damaged_ratio = damaged_pixels / building_pixels if building_pixels else 0.0
    return building_pixels, damaged_pixels, damaged_ratio


def sample_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    confusion = np.zeros((3, 3), dtype=np.float64)
    valid = (target >= 0) & (target < 3)
    indices = target[valid] * 3 + prediction[valid]
    counts = np.bincount(indices, minlength=9).reshape(3, 3)
    confusion += counts

    true_positive = np.diag(confusion)
    total = confusion.sum()
    unions = confusion.sum(axis=1) + confusion.sum(axis=0) - true_positive
    iou = np.divide(
        true_positive,
        unions,
        out=np.full(3, np.nan, dtype=np.float64),
        where=unions > 0,
    )
    return {
        "pixel_accuracy": float(true_positive.sum() / total) if total else 0.0,
        "mean_iou": float(np.nanmean(iou)) if not np.all(np.isnan(iou)) else 0.0,
        "iou_damaged": 0.0 if np.isnan(iou[2]) else float(iou[2]),
    }


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


def render_model_summary(device_name: str, tta_mode: str) -> None:
    st.sidebar.markdown("### Modèle utilisé")
    st.sidebar.write(f"Checkpoint: `{CHECKPOINT_LABEL}`")
    st.sidebar.write(f"Appareil: `{device_name}`")
    st.sidebar.write(f"Taille d'image: `{IMAGE_SIZE}`")
    st.sidebar.write(f"Mode cible: `{TARGET_MODE}`")
    st.sidebar.write(f"Inférence: `{TTA_MODE_LABELS.get(tta_mode, tta_mode)}`")


def validate_dataset_paths() -> None:
    if not DATA_ROOT.exists():
        raise AppError(f"Dossier xBD introuvable : {DATA_ROOT}")
    if not DATA_ROOT.is_dir():
        raise AppError(f"Le chemin xBD n'est pas un dossier : {DATA_ROOT}")
    if not SPLITS_DIR.exists():
        raise AppError(f"Dossier des splits introuvable : {SPLITS_DIR}")


def render_prediction_outputs(
    pre_image: np.ndarray,
    post_image: np.ndarray,
    prediction: np.ndarray,
    title: str,
    tta_mode: str,
    target: np.ndarray | None = None,
) -> None:
    predicted_mask = colorize_mask(prediction)
    prediction_overlay = overlay_prediction(post_image, prediction)

    st.subheader(title)
    st.caption(f"Mode d'inférence : {TTA_MODE_LABELS.get(tta_mode, tta_mode)}")
    render_legend()

    st.markdown("## Résultat principal")
    st.image(
        prediction_overlay,
        caption="Superposition de la prédiction sur l'image après catastrophe",
        width="stretch",
    )

    if target is None:
        st.info("Aucune vérité terrain fournie — inférence uniquement.")

    st.markdown("## Détails de l'inférence")
    detail_cols = st.columns(3)
    detail_cols[0].image(pre_image, caption="Image avant catastrophe", width="stretch")
    detail_cols[1].image(post_image, caption="Image après catastrophe", width="stretch")
    detail_cols[2].image(predicted_mask, caption="Prédiction du modèle", width="stretch")

    if target is not None:
        with st.expander("Vérité terrain et métriques de la paire", expanded=True):
            comparison_cols = st.columns(3)
            comparison_cols[0].image(
                colorize_mask(target),
                caption="Vérité terrain",
                width="stretch",
            )
            comparison_cols[1].image(
                predicted_mask,
                caption="Prédiction du modèle",
                width="stretch",
            )
            comparison_cols[2].image(
                prediction_overlay,
                caption="Superposition",
                width="stretch",
            )

            building_pixels, damaged_pixels, damaged_ratio = prediction_counts(prediction)
            metrics = sample_metrics(prediction, target)
            metric_cols = st.columns(6)
            metric_cols[0].metric("Pixels bâtiments prédits", f"{building_pixels:,}")
            metric_cols[1].metric("Pixels endommagés prédits", f"{damaged_pixels:,}")
            metric_cols[2].metric("Part endommagée prédite", f"{damaged_ratio:.2%}")
            metric_cols[3].metric("Pixel accuracy", f"{metrics['pixel_accuracy']:.3f}")
            metric_cols[4].metric("Mean IoU", f"{metrics['mean_iou']:.3f}")
            metric_cols[5].metric("IoU damaged", f"{metrics['iou_damaged']:.3f}")


def render_dataset_mode(device_name: str, tta_mode: str) -> None:
    try:
        validate_dataset_paths()
    except AppError as exc:
        st.error(str(exc))
        st.stop()

    split_name = st.sidebar.selectbox("Jeu de données", ["train", "val", "test"], index=2)
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
    available_demo_ids = [
        pair_id for pair_id in JALON3_DEMO_PAIR_IDS if pair_id in all_pair_ids_set
    ]

    pair_source = st.sidebar.radio(
        "Sélection des paires",
        ["Toutes les paires disponibles", "Exemples recommandés", "Démo Jalon 3"],
    )
    if pair_source == "Exemples recommandés" and available_recommended_ids:
        pair_ids = available_recommended_ids
    elif pair_source == "Démo Jalon 3" and available_demo_ids:
        pair_ids = available_demo_ids
    else:
        pair_ids = all_pair_ids
        if pair_source != "Toutes les paires disponibles":
            st.sidebar.info(
                "Ces exemples ne sont pas disponibles dans le split courant. "
                "Toutes les paires sont affichées."
            )

    st.sidebar.markdown("#### Paires de démo Jalon 3")
    for pair_id in JALON3_DEMO_PAIR_IDS:
        st.sidebar.caption(pair_id)

    pair_id = st.selectbox("Paire d'images", pair_ids)

    try:
        sample = load_sample(split_name, pair_id)
        model = load_model(str(CHECKPOINT_PATH), device_name)
        prediction = predict(model, sample["image"], device_name, tta_mode)
    except (AppError, XBDDatasetError, RuntimeError, OSError, ValueError) as exc:
        st.error(str(exc))
        st.info(f"Checkpoint attendu : `{CHECKPOINT_PATH}`")
        st.stop()

    image_tensor = sample["image"]
    target = sample["target"].detach().cpu().numpy()
    pred = prediction.numpy()
    pre_image = tensor_to_rgb(image_tensor[:3])
    post_image = tensor_to_rgb(image_tensor[3:6])

    render_prediction_outputs(
        pre_image=pre_image,
        post_image=post_image,
        prediction=pred,
        target=target,
        tta_mode=tta_mode,
        title=f"Paire sélectionnée : `{pair_id}`",
    )

    with st.expander("Détails de l'échantillon"):
        st.write(f"Lignes du split sélectionné : `{len(split_df):,}`")
        st.write(f"Forme du tenseur d'entrée : `{tuple(image_tensor.shape)}`")
        st.write(f"Classes de vérité terrain : `{np.unique(target).tolist()}`")
        st.write(f"Classes prédites : `{np.unique(pred).tolist()}`")


def render_upload_mode(device_name: str, tta_mode: str) -> None:
    st.subheader("Téléverser une paire réelle")
    st.write(
        "Téléversez une image avant catastrophe et une image après catastrophe. "
        "Les deux images seront redimensionnées au format du modèle pour l'inférence."
    )

    upload_cols = st.columns(2)
    pre_file = upload_cols[0].file_uploader(
        "Image avant catastrophe",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        key="upload_pre",
    )
    post_file = upload_cols[1].file_uploader(
        "Image après catastrophe",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        key="upload_post",
    )

    run_clicked = st.button("Lancer l'inférence", type="primary")
    if not run_clicked:
        st.info("Ajoutez les deux images, puis lancez l'inférence.")
        return
    if pre_file is None or post_file is None:
        st.error("Veuillez fournir les deux images : avant et après catastrophe.")
        return

    try:
        image_tensor, pre_image, post_image = uploaded_pair_to_tensor(
            pre_file,
            post_file,
            IMAGE_SIZE,
        )
        model = load_model(str(CHECKPOINT_PATH), device_name)
        prediction = predict(model, image_tensor, device_name, tta_mode)
    except (AppError, RuntimeError, OSError, ValueError) as exc:
        st.error(str(exc))
        st.info(f"Checkpoint attendu : `{CHECKPOINT_PATH}`")
        st.stop()

    render_prediction_outputs(
        pre_image=pre_image,
        post_image=post_image,
        prediction=prediction.numpy(),
        target=None,
        tta_mode=tta_mode,
        title="Résultat d'inférence sur images téléversées",
    )


def main() -> None:
    st.set_page_config(page_title="Aftermath", layout="wide")
    st.title("Aftermath")
    st.caption("Prototype de cartographie automatique des dommages par imagerie satellite")

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    tta_mode_label = st.sidebar.selectbox(
        "Mode d'inférence",
        list(TTA_MODE_OPTIONS.keys()),
        index=2,
        help="La qualité maximale utilise la TTA d4 et peut être plus lente.",
    )
    tta_mode = TTA_MODE_OPTIONS[tta_mode_label]
    render_model_summary(device_name, tta_mode)

    mode = st.sidebar.radio(
        "Mode",
        ["Dataset xBD", "Téléverser une paire réelle"],
        help="Le mode téléversement ne nécessite pas de vérité terrain.",
    )

    if mode == "Dataset xBD":
        render_dataset_mode(device_name, tta_mode)
    else:
        render_upload_mode(device_name, tta_mode)


if __name__ == "__main__":
    main()
