"""Post-processing refinement using SAM2."""

from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class SAMRefiner:
    """Raffine les masques U-Net avec SAM2.

    SAM2 utilise l'image pre-catastrophe pour delimiter precisement
    la forme des batiments, puis fusionne avec les classes U-Net.

    Parameters
    ----------
    checkpoint:
        Chemin vers le fichier .pt de SAM2.
    model_type:
        Config SAM2 yaml.
    device:
        Device torch.
    building_classes:
        Classes considerees comme batiments. 3-class: {1, 2}.
    min_component_pixels:
        Composantes connexes plus petites ignorees.
    """

    def __init__(
        self,
        checkpoint: str,
        model_type: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
        device: torch.device | None = None,
        building_classes: set[int] | None = None,
        min_component_pixels: int = 64,
    ) -> None:
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.building_classes = building_classes or {1, 2}
        self.min_component_pixels = min_component_pixels

        sam2_model = build_sam2(model_type, checkpoint, device=self.device)
        self.predictor = SAM2ImagePredictor(sam2_model)

    def refine_batch(
        self,
        reference_images: torch.Tensor,   # (B, 3, H, W) image pre-catastrophe
        unet_preds: torch.Tensor,          # (B, H, W) masque U-Net
    ) -> tuple[torch.Tensor, torch.Tensor, list[list[np.ndarray]]]:
        """Retourne (masques raffines, masques SAM binaires, bounding boxes par image)."""
        refined_list = []
        sam_binary_list = []
        all_boxes = []
        for img, pred in zip(reference_images, unet_preds):
            refined, sam_binary, boxes = self._refine_single(img, pred)
            refined_list.append(refined)
            sam_binary_list.append(sam_binary)
            all_boxes.append(boxes)
        return torch.stack(refined_list), torch.stack(sam_binary_list), all_boxes

    def _refine_single(
        self,
        reference_image: torch.Tensor,   # (3, H, W) image pre-catastrophe
        unet_pred: torch.Tensor,          # (H, W) masque U-Net
    ) -> tuple[torch.Tensor, torch.Tensor, list[np.ndarray]]:
        img_np = self._to_rgb_numpy(reference_image)
        pred_np = unet_pred.cpu().numpy()
        building_mask = np.isin(pred_np, list(self.building_classes))

        boxes = self._extract_boxes(building_mask)
        if len(boxes) == 0:
            empty_sam = torch.zeros_like(unet_pred, dtype=torch.uint8)
            return unet_pred.clone(), empty_sam, []

        with torch.inference_mode():
            self.predictor.set_image(img_np)
            sam_masks = self._predict_masks(boxes)

        refined = self._merge(pred_np, building_mask, sam_masks)
        # Masque SAM binaire brut : union de tous les masques SAM (1 = batiment selon SAM)
        sam_binary = torch.from_numpy(sam_masks.any(axis=0).astype(np.uint8))
        return refined, sam_binary, boxes

    def _to_rgb_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """(3, H, W) float [0,1] -> (H, W, 3) uint8."""
        img = tensor.cpu()
        if img.is_floating_point():
            img = (img.clamp(0, 1) * 255).byte()
        return img.permute(1, 2, 0).numpy()

    def _extract_boxes(self, binary_mask: np.ndarray) -> list[np.ndarray]:
        """Retourne les bboxes [x1, y1, x2, y2] des composantes connexes."""
        labeled, n = ndimage.label(binary_mask)
        boxes = []
        for label_id in range(1, n + 1):
            component = labeled == label_id
            if component.sum() < self.min_component_pixels:
                continue
            rows = np.where(component.any(axis=1))[0]
            cols = np.where(component.any(axis=0))[0]
            h, w = binary_mask.shape
            pad = 6
            x1 = max(0, int(cols[0]) - pad)
            y1 = max(0, int(rows[0]) - pad)
            x2 = min(w - 1, int(cols[-1]) + pad)
            y2 = min(h - 1, int(rows[-1]) + pad)
            boxes.append(np.array([x1, y1, x2, y2]))
        return boxes

    def _predict_masks(self, boxes: list[np.ndarray]) -> np.ndarray:
        """Appelle SAM2 sur toutes les boxes. Retourne (N, H, W) bool."""
        boxes_np = np.stack(boxes)
        masks, _, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes_np,
            multimask_output=False,
        )
        if masks.ndim == 4:
            masks = masks[:, 0, :, :]
        return masks.astype(bool)

    def _merge(
        self,
        pred_np: np.ndarray,        # (H, W) masque U-Net
        building_mask: np.ndarray,  # (H, W) bool — pixels batiments U-Net
        sam_masks: np.ndarray,      # (N, H, W) bool — masques SAM
    ) -> torch.Tensor:
        """Fusion : SAM definit la forme, U-Net definit la classe pixel par pixel.

        - Pixels dans SAM ET dans U-Net batiment -> classe U-Net preservee
        - Pixels dans SAM mais pas dans U-Net batiment -> background (SAM hors zone U-Net)
        - Pixels hors SAM -> background (faux positifs U-Net supprimes)
        """
        refined = np.zeros_like(pred_np)  # tout background par defaut
        sam_union = sam_masks.any(axis=0)  # (H, W) union de tous les masques SAM

        # Seuls les pixels confirmes par SAM ET classes comme batiment par U-Net sont gardes
        overlap = sam_union & building_mask
        refined[overlap] = pred_np[overlap]

        return torch.from_numpy(refined)