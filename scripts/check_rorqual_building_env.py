"""Check Rorqual Python dependencies for building segmentation campaigns."""

from __future__ import annotations

import importlib
import sys


INSTALL_HINT = (
    "python -m pip install segmentation-models-pytorch==0.5.0 timm==1.0.27"
)


def print_module_version(module_name: str, display_name: str | None = None) -> bool:
    label = display_name or module_name
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        print(f"{label}: MISSING")
        return False

    version = getattr(module, "__version__", "unknown")
    print(f"{label}: {version}")
    return True


def main() -> int:
    print("CrisisMap AI - Rorqual building environment check")
    print("=" * 55)

    torch_ok = print_module_version("torch", "torch")
    if torch_ok:
        import torch

        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            try:
                print(f"CUDA device: {torch.cuda.get_device_name(0)}")
            except Exception as exc:  # pragma: no cover - defensive cluster logging
                print(f"CUDA device name unavailable: {exc}")
        else:
            print(
                "WARNING: CUDA can be False on Rorqual login nodes. "
                "Check CUDA inside an allocated SLURM GPU job."
            )

    timm_ok = print_module_version("timm", "timm")
    smp_ok = print_module_version(
        "segmentation_models_pytorch",
        "segmentation_models_pytorch",
    )

    if not timm_ok or not smp_ok:
        print()
        print("Missing building segmentation dependencies.")
        print("Install manually in the activated venv with:")
        print(f"  {INSTALL_HINT}")
        return 1

    print()
    print("Environment check complete. No packages were installed by this script.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
