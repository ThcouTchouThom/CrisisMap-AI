# Aftermath / CrisisMap AI on Alliance Rorqual

This folder prepares Aftermath / CrisisMap AI for serious Alliance / Rorqual
GPU work. Local 512 training is not the final target. Rorqual should be used
for 1024x1024 training, full-data training, longer runs, hyperparameter sweeps,
and future stronger architectures.

Future stages after the 1024 U-Net baseline and sweep include Siamese U-Net,
ChangeFormer, SegFormer, and hybrid segmentation/classification models. They
are planned, but not implemented here yet.

These scripts use the known working Rorqual defaults:

- account: `def-zonata_gpu`
- partition: `gpubase_bygpu_b2`
- GPU request: `--gres=gpu:h100:1`
- modules: `StdEnv/2023`, `python/3.11`, `gcc`, `arrow/23.0.1`, `cuda`, `opencv/4.13.0`

Account, partition, QoS, module versions, memory, time, and host names may need
adjustment depending on your allocation.

The OpenCV module is intentionally loaded before activating
`~/virtualenvs/crisismap-ai` and before `pip install -r requirements.txt`.
Rorqual resolves `opencv-python` through a Compute Canada dummy wheel that
expects the cluster OpenCV module to be loaded first.

## Expected Layout

Code:

```text
~/work/CrisisMap-AI
```

Large files:

```text
~/scratch/CrisisMap-AI/data
~/scratch/CrisisMap-AI/outputs
~/scratch/CrisisMap-AI/logs
~/scratch/CrisisMap-AI/run_logs
~/scratch/CrisisMap-AI/triton_cache
```

Virtual environment:

```text
~/virtualenvs/crisismap-ai
```

From `~/work/CrisisMap-AI`, setup creates:

```text
data -> ~/scratch/CrisisMap-AI/data
outputs -> ~/scratch/CrisisMap-AI/outputs
```

## Transfer Dataset Archives

Transfer the two archives, not the extracted thousands of image files.

Expected cluster paths:

```text
~/scratch/CrisisMap-AI/data/raw/archives/train_images_labels_targets.tar
~/scratch/CrisisMap-AI/data/raw/archives/xview_geotransforms.json.tgz
```

Create the archive folder on Rorqual:

```bash
mkdir -p ~/scratch/CrisisMap-AI/data/raw/archives
```

Example local Windows PowerShell transfer commands:

```powershell
scp data/raw/archives/train_images_labels_targets.tar tgrjlt2@<RORQUAL_HOST>:~/scratch/CrisisMap-AI/data/raw/archives/
scp data/raw/archives/xview_geotransforms.json.tgz tgrjlt2@<RORQUAL_HOST>:~/scratch/CrisisMap-AI/data/raw/archives/
```

## Clone The Repository

On Rorqual:

```bash
mkdir -p ~/work
cd ~/work
git clone <GITHUB_REPO_URL> CrisisMap-AI
cd ~/work/CrisisMap-AI
```

Manual symlink commands, if ever needed:

```bash
ln -sfn ~/scratch/CrisisMap-AI/data data
ln -sfn ~/scratch/CrisisMap-AI/outputs outputs
```

The setup script normally handles symlinks.

## Setup Environment And Data

From `~/work/CrisisMap-AI`:

```bash
bash slurm/setup_rorqual.sh
```

The setup script:

- creates scratch directories
- creates `data` and `outputs` symlinks
- creates and activates `~/virtualenvs/crisismap-ai`
- exports `TRITON_CACHE_DIR=~/scratch/CrisisMap-AI/triton_cache`
- loads `opencv/4.13.0` before installing Python requirements
- installs `requirements.txt`
- verifies the two archives
- extracts data if needed
- builds `data/processed/xbd_train_index.csv`
- creates full-data splits in `data/processed/splits_full/`

The full split uses all 10 available disasters, `min-nonzero-ratio 0.01`,
`seed 42`, `val-size 0.15`, and `test-size 0.15`.

## Smoke Test

The 512 job is only a technical smoke test for paths, GPU visibility, data
loading, training, checkpoint writing, and logs.

```bash
sbatch slurm/smoke_unet_512.sbatch
```

## Serious 1024 Training

```bash
sbatch slurm/train_unet_full_1024.sbatch
```

This runs the first serious full-data 1024 U-Net baseline:

- image size: `1024`
- batch size: `1`
- epochs: `100`
- loss: `ce-dice`
- class weights: `0.05 1.0 4.0`
- learning rate: `1e-4`

## 1024 Screening Sweep

```bash
sbatch slurm/sweep_unet_1024.sbatch
```

The sweep runs six sequential 1024 U-Net configurations for 50 epochs each on
one H100 GPU. It writes per-run logs to:

```text
~/scratch/CrisisMap-AI/run_logs/
```

It writes SLURM logs to:

```text
~/scratch/CrisisMap-AI/logs/
```

It writes the summary CSV to:

```text
outputs/predictions/unet_1024_sweep_summary.csv
```

Force reruns:

```bash
sbatch --export=ALL,FORCE=1 slurm/sweep_unet_1024.sbatch
```

## Monitor And Cancel

```bash
squeue -u $USER
tail -f ~/scratch/CrisisMap-AI/logs/<logfile>.out
tail -f ~/scratch/CrisisMap-AI/run_logs/<run_log>.log
scancel <jobid>
```

## Retrieve Results

Retrieve checkpoints, metrics, and logs from local Windows PowerShell:

```powershell
scp tgrjlt2@<RORQUAL_HOST>:~/scratch/CrisisMap-AI/outputs/checkpoints/<experiment>/best_unet.pt outputs/checkpoints/<experiment>/
scp tgrjlt2@<RORQUAL_HOST>:~/scratch/CrisisMap-AI/outputs/checkpoints/<experiment>/metrics_history.json outputs/checkpoints/<experiment>/
scp tgrjlt2@<RORQUAL_HOST>:~/scratch/CrisisMap-AI/outputs/predictions/<experiment>_test_metrics.json outputs/predictions/
scp tgrjlt2@<RORQUAL_HOST>:~/scratch/CrisisMap-AI/outputs/predictions/unet_1024_sweep_summary.csv outputs/predictions/
scp tgrjlt2@<RORQUAL_HOST>:~/scratch/CrisisMap-AI/logs/<logfile>.out .
scp tgrjlt2@<RORQUAL_HOST>:~/scratch/CrisisMap-AI/run_logs/<run_log>.log .
```

Retrieve folders:

```powershell
scp -r tgrjlt2@<RORQUAL_HOST>:~/scratch/CrisisMap-AI/outputs/checkpoints/<experiment> outputs/checkpoints/
scp -r tgrjlt2@<RORQUAL_HOST>:~/scratch/CrisisMap-AI/logs .
scp -r tgrjlt2@<RORQUAL_HOST>:~/scratch/CrisisMap-AI/run_logs .
```
