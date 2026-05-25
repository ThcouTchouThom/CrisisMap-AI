from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class SAMRefiner:
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
        post_images: torch.Tensor,
        unet_preds: torch.Tensor,
    ) -> torch.Tensor:
        refined = []
        for img, pred in zip(post_images, unet_preds):
            refined.append(self._refine_single(img, pred))
        return torch.stack(refined)

    def _refine_single(
        self,
        post_image: torch.Tensor,
        unet_pred: torch.Tensor,
    ) -> torch.Tensor:
        img_np = self._to_rgb_numpy(post_image)
        pred_np = unet_pred.cpu().numpy()
        building_mask = np.isin(pred_np, list(self.building_classes))

        boxes = self._extract_boxes(building_mask)
        if len(boxes) == 0:
            return unet_pred.clone()

        with torch.inference_mode():
            self.predictor.set_image(img_np)
            sam_masks = self._predict_masks(boxes)

        return self._merge(pred_np, building_mask, sam_masks)

    def _to_rgb_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        img = tensor.cpu()
        if img.is_floating_point():
            img = (img.clamp(0, 1) * 255).byte()
        return img.permute(1, 2, 0).numpy()

    def _extract_boxes(self, binary_mask: np.ndarray) -> list[np.ndarray]:
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
        pred_np: np.ndarray,
        building_mask: np.ndarray,
        sam_masks: np.ndarray,
    ) -> torch.Tensor:
        refined = pred_np.copy()
        sam_union = sam_masks.any(axis=0)

        # Faux positifs U-Net -> background
        refined[building_mask & ~sam_union] = 0

        # Pixels dans les deux -> classe U-Net préservée
        return torch.from_numpy(refined)
