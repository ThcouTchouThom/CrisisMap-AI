# Alliance / Rorqual

## Environnement

Le développement initial est réalisé localement sous Windows. Les entraînements lourds sont préparés pour le cluster Rorqual de l'Alliance / Calcul Québec avec GPU H100.

Disposition utilisée :

```text
~/work/CrisisMap-AI                  # code
~/scratch/CrisisMap-AI/data          # données
~/scratch/CrisisMap-AI/outputs       # checkpoints et métriques
~/scratch/CrisisMap-AI/logs          # logs SLURM
~/scratch/CrisisMap-AI/run_logs      # logs par expérience
~/virtualenvs/crisismap-ai           # environnement Python
```

Modules utilisés :

```bash
module --force purge
module load StdEnv/2023
module load python/3.11
module load gcc
module load arrow/23.0.1
module load cuda
module load opencv/4.13.0
```

## Scripts SLURM

Les scripts SLURM sont dans `slurm/`. Ils couvrent :

- tests smoke ;
- entraînement U-Net 1024 ;
- sweeps de splits ;
- sweeps augmentation/sampler ;
- longues expériences no-leak.

Les scripts incluent des notifications courriel pour éviter un polling fréquent du scheduler.

## Intérêt pour le jalon 2

Rorqual n'est pas indispensable au jalon 2, mais il montre que le projet est prêt pour :

- entraînements plus longs ;
- résolution 1024 ;
- sweeps d'hyperparamètres ;
- architectures futures plus lourdes.

