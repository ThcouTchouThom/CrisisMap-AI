# Rorqual Submit Notes

Run commands from:

```bash
cd ~/work/CrisisMap-AI
```

## One-Time Setup

```bash
bash slurm/setup_rorqual.sh
```

The setup and job scripts load `opencv/4.13.0` before activating
`~/virtualenvs/crisismap-ai`. Keep that order if editing the module stack,
because `opencv-python` resolves to a Compute Canada wheel that requires the
OpenCV module to be loaded before `pip install -r requirements.txt`.

## Status

```bash
bash slurm/check_cluster_status.sh
```

## Submit Jobs

512 smoke test only:

```bash
sbatch slurm/smoke_unet_512.sbatch
```

Serious 1024 full-data U-Net:

```bash
sbatch slurm/train_unet_full_1024.sbatch
```

1024 screening sweep:

```bash
sbatch slurm/sweep_unet_1024.sbatch
```

Force the sweep to rerun candidates:

```bash
sbatch --export=ALL,FORCE=1 slurm/sweep_unet_1024.sbatch
```

## Monitor

```bash
squeue -u $USER
tail -f ~/scratch/CrisisMap-AI/logs/<file>.out
tail -f ~/scratch/CrisisMap-AI/logs/<file>.err
tail -f ~/scratch/CrisisMap-AI/run_logs/<file>.log
```

## Cancel

```bash
scancel <jobid>
```

## Retrieve Results To Windows

Replace `<RORQUAL_HOST>`, `<experiment>`, and `<file>` as needed:

```powershell
scp tgrjlt2@<RORQUAL_HOST>:~/scratch/CrisisMap-AI/outputs/checkpoints/<experiment>/best_unet.pt outputs/checkpoints/<experiment>/
scp tgrjlt2@<RORQUAL_HOST>:~/scratch/CrisisMap-AI/outputs/checkpoints/<experiment>/metrics_history.json outputs/checkpoints/<experiment>/
scp tgrjlt2@<RORQUAL_HOST>:~/scratch/CrisisMap-AI/outputs/predictions/<experiment>_test_metrics.json outputs/predictions/
scp tgrjlt2@<RORQUAL_HOST>:~/scratch/CrisisMap-AI/outputs/predictions/unet_1024_sweep_summary.csv outputs/predictions/
scp tgrjlt2@<RORQUAL_HOST>:~/scratch/CrisisMap-AI/logs/<file>.out .
scp tgrjlt2@<RORQUAL_HOST>:~/scratch/CrisisMap-AI/logs/<file>.err .
scp tgrjlt2@<RORQUAL_HOST>:~/scratch/CrisisMap-AI/run_logs/<file>.log .
```

Retrieve whole folders:

```powershell
scp -r tgrjlt2@<RORQUAL_HOST>:~/scratch/CrisisMap-AI/outputs/checkpoints/<experiment> outputs/checkpoints/
scp -r tgrjlt2@<RORQUAL_HOST>:~/scratch/CrisisMap-AI/logs .
scp -r tgrjlt2@<RORQUAL_HOST>:~/scratch/CrisisMap-AI/run_logs .
```
