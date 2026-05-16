# Onboarding - CrisisMap AI / Aftermath

Ce document sert de guide rapide pour rejoindre le projet sans lire tout le codebase.

## 1. Vue d'ensemble en 10 lignes

1. Aftermath est un prototype de cartographie automatique des dommages après catastrophe.
2. Le projet utilise le dataset xBD/xView2.
3. Chaque exemple contient une image satellite avant et une image après catastrophe.
4. Le modèle reçoit les deux images RGB concaténées en 6 canaux.
5. La sortie actuelle est un masque à 3 classes : background, no damage, damaged.
6. Le baseline principal est un U-Net de segmentation sémantique.
7. Les données brutes, checkpoints et sorties générées ne sont pas suivis dans Git.
8. Le dépôt contient des scripts pour inspecter, indexer, splitter, entraîner et évaluer.
9. Les entraînements lourds se font sur Rorqual H100 avec SLURM.
10. Le prototype Streamlit permet de visualiser les images, masques et prédictions.

## 2. Structure du dépôt

```text
app/                         # Prototype Streamlit.
configs/                     # Configurations éventuelles.
data/                        # Données locales; les fichiers lourds sont ignorés.
deliverables/                # Livrables de cours.
docs/                        # Documentation pour l'équipe.
notebooks/                   # Exploration.
outputs/                     # Checkpoints, figures, métriques; ignorés.
scripts/                     # Scripts utilitaires locaux.
slurm/                       # Scripts Alliance / Rorqual.
src/crisismap/
  data/                      # Dataset, inspection, indexation, splits.
  evaluation/                # Évaluation et visualisation de prédictions.
  models/                    # U-Net.
  training/                  # Entraînement.
  visualization/             # Figures dataset et métriques.
```

## 3. Cloner et installer localement

```powershell
git clone <URL_DU_REPO>
cd "CrisisMap AI"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Si PowerShell bloque l'activation du venv :

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 4. Placer le dataset

Les archives xBD/xView2 sont partagées hors Git, par exemple via Drive ou Discord.

Place les archives ici :

```text
data/raw/archives/train_images_labels_targets.tar
data/raw/archives/xview_geotransforms.json.tgz
```

Puis lance le setup local :

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_project.ps1
```

Ce script crée les dossiers, installe les dépendances, extrait les archives, inspecte les données, construit l'index et crée les splits de base.

Pour forcer la reconstruction des fichiers traités :

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_project.ps1 -Force
```

## 5. Activer l'environnement Python

À chaque nouvelle session terminal :

```powershell
.\.venv\Scripts\Activate.ps1
```

Vérifier :

```powershell
python --version
pip list
```

Sur Rorqual, le venv est différent :

```bash
source ~/virtualenvs/crisismap-ai/bin/activate
```

## 6. Lancer Streamlit

Depuis la racine du dépôt :

```powershell
streamlit run app/streamlit_app.py
```

Le checkpoint attendu doit être dans `outputs/checkpoints/...`. Les checkpoints ne sont pas dans Git.

## 7. Petit test local

Inspection rapide du dataset :

```powershell
python src/crisismap/data/inspect_xbd.py --root data/raw/xbd/train
```

Test du Dataset PyTorch sur quelques samples :

```powershell
python src/crisismap/data/xbd_dataset.py `
  --root data/raw/xbd/train `
  --split-csv data/processed/splits/train_pairs.csv `
  --num-samples 4
```

Entraînement smoke très court :

```powershell
python src/crisismap/training/train_unet.py `
  --root data/raw/xbd/train `
  --train-csv data/processed/splits/train_pairs.csv `
  --val-csv data/processed/splits/val_pairs.csv `
  --output-dir outputs/checkpoints/local_smoke `
  --image-size 256 `
  --batch-size 1 `
  --epochs 1 `
  --max-train-samples 8 `
  --max-val-samples 4 `
  --target-mode 3-class `
  --loss ce-dice `
  --class-weights 0.05 1.0 4.0
```

## 8. Comprendre les scripts principaux

Données :

- `inspect_xbd.py` : vérifie la structure images/labels/targets.
- `build_xbd_index.py` : crée `data/processed/xbd_train_index.csv`.
- `summarize_xbd_index.py` : résume classes, catastrophes et ratios.
- `create_xbd_splits.py` : crée des splits simples.
- `create_advanced_noleak_train_splits.py` : crée des splits avancés sans fuite.

Modèle et entraînement :

- `xbd_dataset.py` : charge pré/post/mask et crée un tenseur 6 canaux.
- `unet.py` : définit le U-Net.
- `train_unet.py` : entraîne le modèle, avec pertes, augmentation et sampler.
- `evaluate_unet.py` : calcule accuracy, IoU, précision, rappel, F1.
- `predict_unet_sample.py` : visualise une prédiction.

Interface :

- `app/streamlit_app.py` : prototype Aftermath.

Cluster :

- `slurm/setup_rorqual.sh` : préparation côté Rorqual.
- `slurm/*.sbatch` : jobs d'entraînement et sweeps.

## 9. Erreurs fréquentes à éviter

Oublier le venv :

- symptôme : modules introuvables ;
- solution : activer `.venv` localement ou `~/virtualenvs/crisismap-ai` sur Rorqual.

Confondre chemins Windows et Linux :

- Windows local : `data/raw/xbd/train` ou `.\data\raw\xbd\train` ;
- Rorqual : chemins Linux sous `~/work` et `~/scratch`.

Oublier de committer un helper script :

- si un `.sbatch` appelle un script local non committé, Rorqual ne le verra pas après `git pull`.

Créer du data leakage :

- ne jamais entraîner sur un `pair_id` présent dans `data/processed/splits_full/val_pairs.csv` ou `test_pairs.csv` pour les protocoles no-leak ;
- ne jamais augmenter validation/test ;
- ne pas changer le test commun pour comparer deux modèles.

Poller trop souvent Rorqual :

- éviter `watch squeue` ;
- préférer les notifications courriel SLURM ;
- commandes ponctuelles raisonnables : `squeue -u $USER`, `tail -f <log>`.

Commettre des fichiers lourds :

- ne pas versionner `outputs/`, checkpoints `.pt`, images xBD, archives, logs lourds.

## 10. Workflow Git

Avant de commencer :

```powershell
git pull
```

Créer une branche si la modification est non triviale :

```powershell
git checkout -b feature/nom-court
```

Voir les changements :

```powershell
git status
git diff
```

Commits recommandés :

- petits ;
- centrés sur un sujet ;
- avec message clair.

Exemple :

```powershell
git add scripts/ slurm/
git commit -m "Add no-leak augmentation sweep scripts"
git push
```

Avant de commit :

```powershell
git diff --check
```

Ne pas ajouter :

- `data/raw/` ;
- `outputs/checkpoints/` ;
- `outputs/predictions/` volumineux ;
- fichiers `.pt`, `.pth`, `.onnx`.

## 11. Bases Rorqual

Disposition utilisée :

```text
~/work/CrisisMap-AI                 # dépôt Git
~/scratch/CrisisMap-AI/data         # données lourdes
~/scratch/CrisisMap-AI/outputs      # checkpoints, métriques, figures
~/scratch/CrisisMap-AI/logs         # logs SLURM
~/scratch/CrisisMap-AI/run_logs     # logs par expérience
~/virtualenvs/crisismap-ai          # venv Python
```

Modules standards :

```bash
module --force purge
module load StdEnv/2023
module load python/3.11
module load gcc
module load arrow/23.0.1
module load cuda
module load opencv/4.13.0
source ~/virtualenvs/crisismap-ai/bin/activate
```

Préparer le cluster :

```bash
cd ~/work/CrisisMap-AI
bash slurm/setup_rorqual.sh
```

Soumettre un job :

```bash
sbatch slurm/smoke_unet_512.sbatch
sbatch slurm/sweep_unet_1024_noleak_aug_sampler_100epochs_match_hist1000.sbatch
```

Vérifier un job sans poller agressivement :

```bash
squeue -u $USER
tail -f ~/scratch/CrisisMap-AI/logs/<fichier>.out
```

Annuler un job :

```bash
scancel <jobid>
```

Récupérer les résultats depuis Windows se fait avec `scp`, en copiant surtout les métriques JSON/CSV, figures utiles et checkpoints choisis.

