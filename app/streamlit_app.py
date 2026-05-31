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
PROJECT_SCRIPTS = PROJECT_ROOT / "scripts"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))
if str(PROJECT_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_SCRIPTS))

from crisismap.data.xbd_dataset import (  # noqa: E402
    XBDDatasetError,
    XBDPairDataset as XBDDataset,
)
from crisismap.models.unet import UNet  # noqa: E402
from train_building_segmentation import (  # noqa: E402
    BuildingTrainingError,
    build_model as build_building_model,
    clean_state_dict as clean_building_state_dict,
    input_channels as building_input_channels,
    normalize_logits as normalize_building_logits,
)


DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "xbd" / "train"
SPLITS_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
    / "unet_1024_long250_noleak_match_hist_all_aug-safe_sampler-damage-sqrt-alpha4_250epochs"
    / "best_unet_portable.pt"
)
BUILDING_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
    / "b100_d_full_pre_unetplusplus_effb4_sampler8_focal_tversky"
    / "best_building_portable.pt"
)

IMAGE_SIZE = 1024
TARGET_MODE = "3-class"
BASE_CHANNELS = 32
BUILDING_MODEL_NAME = "unetplusplus_effb4"
BUILDING_INPUT_MODE = "pre"
BUILDING_COMPONENT_CONNECTIVITY = 8
CHECKPOINT_LABEL = (
    "unet_1024_long250_noleak_match_hist_all_aug-safe_sampler-damage-sqrt-alpha4_250epochs/"
    "best_unet_portable.pt"
)
BUILDING_CHECKPOINT_LABEL = (
    "b100_d_full_pre_unetplusplus_effb4_sampler8_focal_tversky/"
    "best_building_portable.pt"
)
INFERENCE_MODE_OPTIONS = {
    "Damage only": {
        "damage_tta": "none",
        "use_building": False,
        "description": "U-Net sans TTA, pour les demos rapides.",
    },
    "Damage + TTA d4": {
        "damage_tta": "d4",
        "use_building": False,
        "description": "Pipeline damage officiel avec TTA d4.",
    },
    "Damage + TTA d4 + building component majority": {
        "damage_tta": "d4",
        "use_building": True,
        "description": "Pipeline downstream actuel avec masque batiment predit.",
    },
}

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


@st.cache_resource(show_spinner="Chargement du modele batiment...")
def load_building_model(
    checkpoint_path: str,
    device_name: str,
    model_name: str,
    input_mode: str,
) -> torch.nn.Module:
    checkpoint = resolve_project_path(checkpoint_path)
    if not checkpoint.exists():
        raise AppError(f"Checkpoint batiment introuvable : {checkpoint}")
    if not checkpoint.is_file():
        raise AppError(f"Le checkpoint batiment n'est pas un fichier : {checkpoint}")

    device = torch.device(device_name)
    try:
        model, _ = build_building_model(
            model_name,
            building_input_channels(input_mode),
            device,
        )
        loaded = load_checkpoint_file(checkpoint, device)
        if isinstance(loaded, dict) and "model_state_dict" in loaded:
            state_dict = loaded["model_state_dict"]
        else:
            state_dict = loaded
        model.load_state_dict(clean_building_state_dict(state_dict))
    except (BuildingTrainingError, RuntimeError, OSError) as exc:
        raise AppError(
            "Impossible de charger le modele batiment. Verifiez le checkpoint "
            "et les dependances segmentation-models-pytorch/timm."
        ) from exc

    model.eval()
    return model


def resolve_project_path(path_text: str) -> Path:
    path = Path(path_text.strip()).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


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
    inference_mode: str,
    building_checkpoint_path: str,
    building_threshold: float,
) -> torch.Tensor:
    device = torch.device(device_name)
    options = INFERENCE_MODE_OPTIONS[inference_mode]
    tta_mode = str(options["damage_tta"])
    image_batch = image.unsqueeze(0).to(device)
    logits = predict_logits_tta(
        model=model,
        image=image_batch,
        tta_mode=tta_mode,
    )
    raw_prediction = torch.argmax(logits, dim=1)
    if not bool(options["use_building"]):
        return raw_prediction.squeeze(0).cpu()

    building_model = load_building_model(
        building_checkpoint_path,
        device_name,
        BUILDING_MODEL_NAME,
        BUILDING_INPUT_MODE,
    )
    building_input = select_building_input(image_batch, BUILDING_INPUT_MODE)
    building_logits = normalize_building_logits(building_model(building_input))
    building_probs = torch.sigmoid(building_logits).squeeze(1)
    building_mask = building_probs >= building_threshold
    post_processed = component_majority_batch(
        raw_prediction,
        building_mask,
        connectivity=BUILDING_COMPONENT_CONNECTIVITY,
        device=device,
    )
    return post_processed.squeeze(0).cpu()


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


def select_building_input(images: torch.Tensor, input_mode: str) -> torch.Tensor:
    if input_mode == "pre":
        return images[:, :3]
    if input_mode == "post":
        return images[:, 3:6]
    if input_mode == "pre-post":
        return images
    raise AppError(f"Mode d'entree batiment non pris en charge : {input_mode}")


def component_majority_batch(
    raw_predictions: torch.Tensor,
    building_masks: torch.Tensor,
    connectivity: int,
    device: torch.device,
) -> torch.Tensor:
    outputs = []
    for prediction_tensor, mask_tensor in zip(raw_predictions, building_masks):
        prediction = prediction_tensor.detach().cpu().numpy().astype(np.int16, copy=False)
        building_mask = mask_tensor.detach().cpu().numpy().astype(bool, copy=False)
        outputs.append(component_majority_single(prediction, building_mask, connectivity))
    return torch.from_numpy(np.stack(outputs, axis=0)).to(device=device, dtype=torch.long)


def component_majority_single(
    raw_prediction: np.ndarray,
    building_mask: np.ndarray,
    connectivity: int,
) -> np.ndarray:
    labels, component_count = label_connected_components(building_mask, connectivity)
    output = np.zeros_like(raw_prediction, dtype=np.int64)
    for component_id in range(1, component_count + 1):
        component_mask = labels == component_id
        component_predictions = raw_prediction[component_mask]
        no_damage_count = int(np.count_nonzero(component_predictions == 1))
        damaged_count = int(np.count_nonzero(component_predictions == 2))
        component_class = 2 if damaged_count > no_damage_count else 1
        output[component_mask] = component_class
    return output


def label_connected_components(
    mask: np.ndarray,
    connectivity: int,
) -> tuple[np.ndarray, int]:
    mask = np.asarray(mask, dtype=bool)
    structure = (
        np.ones((3, 3), dtype=np.uint8)
        if connectivity == 8
        else np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    )

    try:
        from scipy import ndimage  # type: ignore

        labels, count = ndimage.label(mask, structure=structure)
        return labels.astype(np.int32, copy=False), int(count)
    except ImportError:
        pass

    try:
        from skimage.measure import label as skimage_label  # type: ignore

        sk_connectivity = 2 if connectivity == 8 else 1
        labels = skimage_label(mask, connectivity=sk_connectivity, background=0)
        return labels.astype(np.int32, copy=False), int(labels.max())
    except ImportError:
        return label_connected_components_fallback(mask, connectivity)


def label_connected_components_fallback(
    mask: np.ndarray,
    connectivity: int,
) -> tuple[np.ndarray, int]:
    labels = np.zeros(mask.shape, dtype=np.int32)
    height, width = mask.shape
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        neighbors.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])

    current_label = 0
    for row in range(height):
        for col in range(width):
            if not mask[row, col] or labels[row, col] != 0:
                continue
            current_label += 1
            stack = [(row, col)]
            labels[row, col] = current_label
            while stack:
                y, x = stack.pop()
                for dy, dx in neighbors:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        if mask[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = current_label
                            stack.append((ny, nx))
    return labels, current_label


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


def render_model_summary(
    device_name: str,
    inference_mode: str,
    building_checkpoint_path: str,
    building_threshold: float,
) -> None:
    options = INFERENCE_MODE_OPTIONS[inference_mode]
    st.sidebar.markdown("### Modèle utilisé")
    st.sidebar.write(f"Checkpoint damage: `{CHECKPOINT_LABEL}`")
    st.sidebar.write(f"Appareil: `{device_name}`")
    st.sidebar.write(f"Taille d'image: `{IMAGE_SIZE}`")
    st.sidebar.write(f"Mode cible: `{TARGET_MODE}`")
    st.sidebar.write(f"Pipeline: `{inference_mode}`")
    st.sidebar.write(f"TTA damage: `{options['damage_tta']}`")
    if bool(options["use_building"]):
        st.sidebar.write(f"Checkpoint bâtiment: `{building_checkpoint_path}`")
        st.sidebar.write(f"Modèle bâtiment: `{BUILDING_MODEL_NAME}` / `{BUILDING_INPUT_MODE}`")
        st.sidebar.write(f"Seuil bâtiment: `{building_threshold:.2f}`")


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
    inference_mode: str,
    target: np.ndarray | None = None,
) -> None:
    predicted_mask = colorize_mask(prediction)
    prediction_overlay = overlay_prediction(post_image, prediction)

    st.subheader(title)
    st.caption(f"Pipeline d'inférence : {inference_mode}")
    if bool(INFERENCE_MODE_OPTIONS[inference_mode]["use_building"]):
        st.warning(
            "Le post-traitement bâtiment améliore les métriques globales actuelles, "
            "mais il peut encore supprimer de vrais pixels endommagés si la "
            "segmentation bâtiment les manque."
        )
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


def render_dataset_mode(
    device_name: str,
    inference_mode: str,
    building_checkpoint_path: str,
    building_threshold: float,
) -> None:
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
        prediction = predict(
            model,
            sample["image"],
            device_name,
            inference_mode,
            building_checkpoint_path,
            building_threshold,
        )
    except (
        AppError,
        XBDDatasetError,
        BuildingTrainingError,
        RuntimeError,
        OSError,
        ValueError,
    ) as exc:
        st.error(str(exc))
        st.info(f"Checkpoint attendu : `{CHECKPOINT_PATH}`")
        if bool(INFERENCE_MODE_OPTIONS[inference_mode]["use_building"]):
            st.info(f"Checkpoint bâtiment attendu : `{building_checkpoint_path}`")
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
        inference_mode=inference_mode,
        title=f"Paire sélectionnée : `{pair_id}`",
    )

    with st.expander("Détails de l'échantillon"):
        st.write(f"Lignes du split sélectionné : `{len(split_df):,}`")
        st.write(f"Forme du tenseur d'entrée : `{tuple(image_tensor.shape)}`")
        st.write(f"Classes de vérité terrain : `{np.unique(target).tolist()}`")
        st.write(f"Classes prédites : `{np.unique(pred).tolist()}`")


def render_upload_mode(
    device_name: str,
    inference_mode: str,
    building_checkpoint_path: str,
    building_threshold: float,
) -> None:
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
        prediction = predict(
            model,
            image_tensor,
            device_name,
            inference_mode,
            building_checkpoint_path,
            building_threshold,
        )
    except (
        AppError,
        BuildingTrainingError,
        RuntimeError,
        OSError,
        ValueError,
    ) as exc:
        st.error(str(exc))
        st.info(f"Checkpoint attendu : `{CHECKPOINT_PATH}`")
        if bool(INFERENCE_MODE_OPTIONS[inference_mode]["use_building"]):
            st.info(f"Checkpoint bâtiment attendu : `{building_checkpoint_path}`")
        st.stop()

    render_prediction_outputs(
        pre_image=pre_image,
        post_image=post_image,
        prediction=prediction.numpy(),
        target=None,
        inference_mode=inference_mode,
        title="Résultat d'inférence sur images téléversées",
    )


def main() -> None:
    st.set_page_config(page_title="Aftermath", layout="wide")
    st.title("Aftermath")
    st.caption("Prototype de cartographie automatique des dommages par imagerie satellite")

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    building_checkpoint_default = str(BUILDING_CHECKPOINT_PATH)
    building_checkpoint_exists = BUILDING_CHECKPOINT_PATH.exists()
    inference_modes = list(INFERENCE_MODE_OPTIONS.keys())
    default_mode = (
        "Damage + TTA d4 + building component majority"
        if building_checkpoint_exists
        else "Damage + TTA d4"
    )
    inference_mode = st.sidebar.selectbox(
        "Pipeline d'inférence",
        inference_modes,
        index=inference_modes.index(default_mode),
        help=(
            "Le mode downstream ajoute une segmentation bâtiment prédite et "
            "une décision majoritaire par composante."
        ),
    )
    st.sidebar.caption(str(INFERENCE_MODE_OPTIONS[inference_mode]["description"]))
    building_checkpoint_path = st.sidebar.text_input(
        "Checkpoint bâtiment",
        value=building_checkpoint_default,
        help=f"Checkpoint recommandé : {BUILDING_CHECKPOINT_LABEL}",
    )
    building_threshold = st.sidebar.slider(
        "Seuil bâtiment",
        min_value=0.10,
        max_value=0.90,
        value=0.60,
        step=0.05,
    )
    if (
        inference_mode == "Damage + TTA d4 + building component majority"
        and not resolve_project_path(building_checkpoint_path).exists()
    ):
        st.sidebar.warning(
            "Checkpoint bâtiment introuvable : le mode TTA d4 seul reste disponible."
        )

    render_model_summary(
        device_name,
        inference_mode,
        building_checkpoint_path,
        building_threshold,
    )

    mode = st.sidebar.radio(
        "Mode",
        ["Dataset xBD", "Téléverser une paire réelle"],
        help="Le mode téléversement ne nécessite pas de vérité terrain.",
    )

    if mode == "Dataset xBD":
        render_dataset_mode(
            device_name,
            inference_mode,
            building_checkpoint_path,
            building_threshold,
        )
    else:
        render_upload_mode(
            device_name,
            inference_mode,
            building_checkpoint_path,
            building_threshold,
        )


if __name__ == "__main__":
    main()
