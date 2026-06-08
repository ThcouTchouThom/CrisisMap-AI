"""Factory for damage segmentation architectures used in Axis 2 experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

from crisismap.models.siamese_unet import (
    SiameseUNetAbsDiff,
    SiameseUNetAbsSigned,
    SiameseUNetAbsSignedProduct,
    SiameseUNetAttention,
    SiameseUNetGatedFusion,
    SiameseUNetSharedEncoder,
)
from crisismap.models.unet import UNet


@dataclass(frozen=True)
class DamageModelSpec:
    name: str
    family: str
    description: str


SUPPORTED_DAMAGE_MODELS: dict[str, DamageModelSpec] = {
    "local_unet_existing": DamageModelSpec(
        name="local_unet_existing",
        family="local_unet",
        description="Existing compact local U-Net with 6-channel input.",
    ),
    "siamese_unet_shared_encoder": DamageModelSpec(
        name="siamese_unet_shared_encoder",
        family="siamese",
        description="Shared-encoder Siamese U-Net with multi-level change fusion.",
    ),
    "siamese_unet_absdiff": DamageModelSpec(
        name="siamese_unet_absdiff",
        family="siamese",
        description="Siamese U-Net using post features plus absolute feature difference.",
    ),
    "siamese_unet_abs_signed": DamageModelSpec(
        name="siamese_unet_abs_signed",
        family="siamese",
        description="Siamese U-Net with absolute and signed feature differences.",
    ),
    "siamese_unet_abs_signed_product": DamageModelSpec(
        name="siamese_unet_abs_signed_product",
        family="siamese",
        description="Siamese U-Net with abs, signed, and feature-product fusion.",
    ),
    "siamese_unet_gated_fusion": DamageModelSpec(
        name="siamese_unet_gated_fusion",
        family="siamese",
        description="Siamese U-Net with a learned abs-change gate.",
    ),
    "siamese_unet_attention": DamageModelSpec(
        name="siamese_unet_attention",
        family="siamese",
        description="Siamese U-Net with channel attention after fusion.",
    ),
    "siamese_unet_attention_base48": DamageModelSpec(
        name="siamese_unet_attention_base48",
        family="siamese",
        description="Siamese attention U-Net with base_channels=48.",
    ),
    "siamese_unet_attention_base64": DamageModelSpec(
        name="siamese_unet_attention_base64",
        family="siamese",
        description="Siamese attention U-Net with base_channels=64.",
    ),
    "siamese_unet_base48": DamageModelSpec(
        name="siamese_unet_base48",
        family="siamese",
        description="Shared-encoder Siamese U-Net with base_channels=48.",
    ),
    "siamese_unet_base64": DamageModelSpec(
        name="siamese_unet_base64",
        family="siamese",
        description="Shared-encoder Siamese U-Net with base_channels=64.",
    ),
    "smp_unet_effb3_6ch": DamageModelSpec(
        name="smp_unet_effb3_6ch",
        family="smp",
        description="SMP U-Net, EfficientNet-B3 encoder, 6-channel input.",
    ),
    "smp_unet_resnet50_6ch": DamageModelSpec(
        name="smp_unet_resnet50_6ch",
        family="smp",
        description="SMP U-Net, ResNet-50 encoder, 6-channel input.",
    ),
    "smp_deeplabv3plus_resnet50_6ch": DamageModelSpec(
        name="smp_deeplabv3plus_resnet50_6ch",
        family="smp",
        description="SMP DeepLabV3+, ResNet-50 encoder, 6-channel input.",
    ),
    "smp_deeplabv3plus_effb3_6ch": DamageModelSpec(
        name="smp_deeplabv3plus_effb3_6ch",
        family="smp",
        description="SMP DeepLabV3+, EfficientNet-B3 encoder, 6-channel input.",
    ),
}

MODEL_ALIASES = {
    "unet": "local_unet_existing",
    "siamese_unet_simple": "siamese_unet_shared_encoder",
}


class DamageModelFactoryError(RuntimeError):
    """Raised when a requested damage architecture cannot be created."""


def canonical_model_name(model_name: str) -> str:
    return MODEL_ALIASES.get(model_name, model_name)


def supported_damage_model_names() -> list[str]:
    return sorted(SUPPORTED_DAMAGE_MODELS)


def damage_model_metadata(model_name: str) -> dict[str, str]:
    canonical_name = canonical_model_name(model_name)
    spec = SUPPORTED_DAMAGE_MODELS.get(canonical_name)
    if spec is None:
        raise DamageModelFactoryError(
            f"Unsupported damage model '{model_name}'. Supported models: "
            + ", ".join(supported_damage_model_names())
        )
    return {
        "requested_model": model_name,
        "canonical_model": canonical_name,
        "family": spec.family,
        "description": spec.description,
    }


def create_damage_model(
    model_name: str,
    num_classes: int = 3,
    in_channels: int = 6,
    base_channels: int = 32,
) -> nn.Module:
    """Create a damage segmentation model by name.

    SMP architectures intentionally use encoder_weights=None because the 6-channel
    pre/post input does not match off-the-shelf RGB pretrained stems.
    """

    canonical_name = canonical_model_name(model_name)
    if canonical_name not in SUPPORTED_DAMAGE_MODELS:
        raise DamageModelFactoryError(
            f"Unsupported damage model '{model_name}'. Supported models: "
            + ", ".join(supported_damage_model_names())
        )

    if canonical_name == "local_unet_existing":
        return UNet(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
        )

    if canonical_name == "siamese_unet_shared_encoder":
        return SiameseUNetSharedEncoder(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
        )

    if canonical_name == "siamese_unet_absdiff":
        return SiameseUNetAbsDiff(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
        )

    if canonical_name == "siamese_unet_abs_signed":
        return SiameseUNetAbsSigned(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
        )

    if canonical_name == "siamese_unet_abs_signed_product":
        return SiameseUNetAbsSignedProduct(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
        )

    if canonical_name == "siamese_unet_gated_fusion":
        return SiameseUNetGatedFusion(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
        )

    if canonical_name == "siamese_unet_attention":
        return SiameseUNetAttention(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
        )

    if canonical_name == "siamese_unet_attention_base48":
        return SiameseUNetAttention(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=48,
        )

    if canonical_name == "siamese_unet_attention_base64":
        return SiameseUNetAttention(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=64,
        )

    if canonical_name == "siamese_unet_base48":
        return SiameseUNetSharedEncoder(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=48,
        )

    if canonical_name == "siamese_unet_base64":
        return SiameseUNetSharedEncoder(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=64,
        )

    return create_smp_damage_model(
        model_name=canonical_name,
        num_classes=num_classes,
        in_channels=in_channels,
    )


def create_smp_damage_model(
    model_name: str,
    num_classes: int,
    in_channels: int,
) -> nn.Module:
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise DamageModelFactoryError(
            "The requested model requires segmentation-models-pytorch. "
            "Install it with: python -m pip install "
            "segmentation-models-pytorch==0.5.0 timm==1.0.27"
        ) from exc

    if model_name == "smp_unet_effb3_6ch":
        return smp.Unet(
            encoder_name="efficientnet-b3",
            encoder_weights=None,
            in_channels=in_channels,
            classes=num_classes,
        )
    if model_name == "smp_unet_resnet50_6ch":
        return smp.Unet(
            encoder_name="resnet50",
            encoder_weights=None,
            in_channels=in_channels,
            classes=num_classes,
        )
    if model_name == "smp_deeplabv3plus_resnet50_6ch":
        return smp.DeepLabV3Plus(
            encoder_name="resnet50",
            encoder_weights=None,
            in_channels=in_channels,
            classes=num_classes,
        )
    if model_name == "smp_deeplabv3plus_effb3_6ch":
        return smp.DeepLabV3Plus(
            encoder_name="efficientnet-b3",
            encoder_weights=None,
            in_channels=in_channels,
            classes=num_classes,
        )

    raise DamageModelFactoryError(f"Unsupported SMP damage model: {model_name}")
