# CrisisMap AI

## Project Overview

CrisisMap AI is an academic prototype for disaster damage assessment from paired satellite imagery. It analyzes pre-disaster and post-disaster RGB image pairs from the xBD/xView2 dataset and produces pixel-level damage maps.

The current MVP is a 3-class semantic segmentation baseline:

- `0`: background
- `1`: no damage
- `2`: damaged

## Dataset

The project uses the xBD/xView2 training set. Raw data is expected locally under:

```text
data/raw/xbd/
```

The xBD archive, extracted satellite images, labels, generated masks, model checkpoints, and prediction outputs are not stored in GitHub. Keep raw imagery under `data/raw/`, derived CSVs under `data/processed/`, and generated artifacts under `outputs/`.

## Setup For Teammates

Clone the repository, then place the externally shared xBD/xView2 archives in:

```text
data/raw/archives/
```

Expected files:

- `data/raw/archives/train_images_labels_targets.tar`
- `data/raw/archives/xview_geotransforms.json.tgz`

Run the Windows PowerShell setup script from the project root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_project.ps1
```

To rebuild processed index and split CSV files:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_project.ps1 -Force
```

The script creates `.venv`, installs dependencies, extracts the local archives, validates the dataset, builds `data/processed/xbd_train_index.csv`, and creates split CSVs under `data/processed/splits/`.

If you received a trained checkpoint, place it at:

```text
outputs/checkpoints/unet_baseline_512_v2_30epochs/best_unet.pt
```

To have the SAM2 checkpoint for mask refinement:

```powershell
New-Item -ItemType Directory -Force -Path "src\models\sam"
Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt" -OutFile "models\sam\sam_vit_b_01ec64.pth"
```

SAM2 is optional. The Streamlit app works without it but the "Activer SAM" toggle in the sidebar will be disabled.

Run the Streamlit prototype:

```powershell
streamlit run app/streamlit_app.py
```

## MVP Approach

Each sample combines:

- pre-disaster RGB image
- post-disaster RGB image
- target damage mask

The model input is a 6-channel tensor formed by concatenating pre-disaster RGB and post-disaster RGB images. The baseline model is a lightweight U-Net with `7,763,971` trainable parameters.
A SAM2 (Segment Anything Model 2) post-processing step is optionally applied after U-Net inference to refine building contours using the post-disaster image as visual reference.

## Current Pipeline

1. Inspect the extracted xBD/xView2 folder structure.
2. Visualize image pairs, labels, and target masks.
3. Build a CSV index of valid image pairs.
4. Summarize class imbalance and disaster distribution.
5. Create train/val/test split CSVs.
6. Train a U-Net baseline.
7. Evaluate the checkpoint on validation or test splits.
8. Generate prediction visualizations and training metric plots.

## Results

Training subset:

- train: `565` pairs
- val: `122` pairs
- test: `122` pairs
- disasters: `hurricane-harvey`, `hurricane-michael`, `palu-tsunami`, `santa-rosa-wildfire`

Best training run:

- image size: `512`
- batch size: `2`
- epochs: `30`
- best validation mean IoU: `0.6322` at epoch `29`

Test metrics:

- pixel accuracy: `0.9175`
- mean IoU: `0.6257`
- IoU background: `0.9297`
- IoU no damage: `0.5602`
- IoU damaged: `0.3870`
- F1 damaged: `0.5581`

## How To Run Key Scripts

Inspect the dataset:

```powershell
python .\src\crisismap\data\inspect_xbd.py --root ".\data\raw\xbd\train"
```

Visualize a sample:

```powershell
python .\src\crisismap\visualization\visualize_xbd_sample.py --root ".\data\raw\xbd\train" --mode 3-class
```

Build the dataset index:

```powershell
python .\src\crisismap\data\build_xbd_index.py --root ".\data\raw\xbd\train" --output ".\data\processed\xbd_train_index.csv"
```

Summarize the index:

```powershell
python .\src\crisismap\data\summarize_xbd_index.py --index ".\data\processed\xbd_train_index.csv"
```

Create split CSVs:

```powershell
python .\src\crisismap\data\create_xbd_splits.py --index ".\data\processed\xbd_train_index.csv" --output-dir ".\data\processed\splits"
```

Smoke-test the PyTorch dataset:

```powershell
python .\src\crisismap\data\xbd_dataset.py --root ".\data\raw\xbd\train" --split-csv ".\data\processed\splits\train_pairs.csv" --num-samples 4
```

Train the baseline U-Net:

```powershell
python .\src\crisismap\training\train_unet.py --root ".\data\raw\xbd\train" --train-csv ".\data\processed\splits\train_pairs.csv" --val-csv ".\data\processed\splits\val_pairs.csv" --output-dir ".\outputs\checkpoints\unet_baseline_512_v2_30epochs" --image-size 512 --batch-size 2 --epochs 30 --target-mode 3-class
```

Evaluate a checkpoint:

```powershell
python .\src\crisismap\evaluation\evaluate_unet.py --root ".\data\raw\xbd\train" --split-csv ".\data\processed\splits\test_pairs.csv" --checkpoint ".\outputs\checkpoints\unet_baseline_512_v2_30epochs\best_unet.pt" --output ".\outputs\predictions\unet_test_metrics.json" --image-size 512 --target-mode 3-class
```

Visualize one prediction:

```powershell
python .\src\crisismap\evaluation\predict_unet_sample.py --root ".\data\raw\xbd\train" --split-csv ".\data\processed\splits\test_pairs.csv" --checkpoint ".\outputs\checkpoints\unet_baseline_512_v2_30epochs\best_unet.pt" --image-size 512 --target-mode 3-class
```

Plot training metrics:

```powershell
python .\src\crisismap\visualization\plot_training_metrics.py --metrics ".\outputs\checkpoints\unet_baseline_512_v2_30epochs\metrics_history.json" --output-dir ".\outputs\figures\training_metrics_512_v2"
```

## Repository Structure

```text
CrisisMap AI/
  app/                         # Prototype UI or demo entry points.
  configs/                     # Experiment and path configuration files.
  data/
    raw/                       # Local xBD/xView2 archive extraction; not committed.
    processed/                 # Local indexes and split CSVs.
    samples/                   # Small optional demo samples only.
  notebooks/                   # Exploratory analysis notebooks.
  outputs/
    checkpoints/               # Local model checkpoints; not committed.
    figures/                   # Local plots and visual summaries.
    predictions/               # Local metrics and prediction artifacts.
  src/crisismap/
    data/                      # Inspection, indexing, splitting, and dataset code.
    evaluation/                # Evaluation and prediction scripts.
    models/                    # U-Net model definition.
      sam/                       # SAM2 checkpoint; not committed (download separately).
    training/                  # U-Net training script.
    visualization/             # Dataset and metrics visualization scripts.
```

## Next Steps

- Improve damaged-class performance through stronger class balancing or loss functions.
- Add data augmentation for disaster imagery.
- Compare against stronger segmentation backbones.
- Add qualitative error analysis for damaged-building false negatives.
- Package a small demo workflow using saved predictions and figures.