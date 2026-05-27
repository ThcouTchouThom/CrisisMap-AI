# Onboarding - CrisisMap AI / Aftermath

Ce document sert de guide rapide pour rejoindre le projet sans lire tout le codebase.

## 1. Vue d'ensemble en 10 lignes

1. Aftermath est un prototype de cartographie automatique des dommages après catastrophe.
2. Le projet utilise le dataset xBD/xView2.
3. Chaque exemple contient une image satellite avant et une image après catastrophe.
4. Le modèle damage reçoit les deux images RGB concaténées en 6 canaux.
5. La sortie damage actuelle est un masque à 3 classes : fond, bâtiment non endommagé, bâtiment endommagé.
6. Le baseline principal est un U-Net de segmentation sémantique.
7. Le protocole courant utilise une validation et un test communs no-leak.
8. Une branche building-only évalue la segmentation binaire fond/bâtiment.
9. Les entraînements lourds se font sur Rorqual H100 avec SLURM.
10. Le prototype Streamlit alpha accepte aussi une paire réelle téléversée pour inférence.

## 2. Structure du dépôt

```text
app/                         # Prototype Streamlit.
configs/                     # Notes et configurations éventuelles.
data/                        # Données locales; les fichiers lourds sont ignorés.
deliverables/                # Sources légères de livrables de cours.
docs/                        # Documentation pour l'équipe.
outputs/                     # Checkpoints, figures, métriques; ignorés.
scripts/                     # Scripts utilitaires, splits, évaluations, building-only.
slurm/                       # Scripts Alliance / Rorqual.
src/crisismap/
  data/                      # Dataset, inspection, indexation, splits.
  evaluation/                # Évaluation et visualisation de prédictions.
  models/                    # U-Net.
  training/                  # Entraînement damage.
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

Le prototype alpha possède deux modes :

- mode dataset xBD : sélection d'un split et d'une paire connue, avec vérité terrain et métriques locales ;
- mode téléversement : ajout d'une paire réelle pré/post, inférence uniquement, sans métriques faute de vérité terrain.

Le checkpoint attendu n'est pas dans Git. Pour la démo damage actuelle, il doit être restauré ici :

```text
outputs/checkpoints/unet_1024_ce_dice_w005_1_4_noleak_match_hist1000_bs2_250epochs/best_unet_portable.pt
```

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

- `src/crisismap/data/inspect_xbd.py` : vérifie la structure images/labels/targets.
- `src/crisismap/data/build_xbd_index.py` : crée `data/processed/xbd_train_index.csv`.
- `src/crisismap/data/summarize_xbd_index.py` : résume classes, catastrophes et ratios.
- `src/crisismap/data/create_xbd_splits.py` : crée des splits simples.
- `scripts/create_noleak_common_eval_splits.py` : crée des splits avec validation/test communs.
- `scripts/create_advanced_noleak_train_splits.py` : crée des splits avancés sans fuite.

Modèles et évaluation :

- `src/crisismap/data/xbd_dataset.py` : charge pré/post/mask et crée le tenseur 6 canaux.
- `src/crisismap/models/unet.py` : définit le U-Net damage.
- `src/crisismap/training/train_unet.py` : entraîne le modèle damage avec pertes, augmentation et sampler.
- `src/crisismap/evaluation/evaluate_unet.py` : calcule accuracy, IoU, précision, rappel, F1.
- `scripts/evaluate_oracle_building_mask_gain.py` : mesure le gain théorique d'un masque bâtiment parfait.
- `scripts/train_building_segmentation.py` : entraîne la segmentation binaire fond/bâtiment.
- `scripts/evaluate_building_segmentation.py` : évalue un checkpoint building-only.
- `scripts/evaluate_damage_with_predicted_building_mask.py` : compare la prédiction damage brute, le masque bâtiment prédit et les oracles.
- `scripts/rebuild_noleak_aug_sampler_summary.py` : reconstruit le résumé de la campagne augmentation/sampler damage.
- `scripts/rebuild_building100_summary.py` : reconstruit le résumé de la campagne Building100.

Interface :

- `app/streamlit_app.py` : prototype Aftermath.

Cluster :

- `slurm/setup_rorqual.sh` : préparation côté Rorqual.
- `slurm/*.sbatch` : jobs d'entraînement, sweeps et évaluations.
- `slurm/submit_long250_aug_sampler_campaign.sh` : soumet les longs runs damage sélectionnés.
- `slurm/submit_building100_sweep_v1.sh` : soumet la campagne large building-only.

## 9. Erreurs fréquentes à éviter

Oublier le venv :

- symptôme : modules introuvables ;
- solution : activer `.venv` localement ou `~/virtualenvs/crisismap-ai` sur Rorqual.

Confondre chemins Windows et Linux :

- Windows local : `data/raw/xbd/train` ou `.\data\raw\xbd\train` ;
- Rorqual : chemins Linux sous `~/work` et `~/scratch`.

Oublier que les checkpoints ne sont pas dans Git :

- le dépôt contient le code et les scripts ;
- les fichiers `.pt`, `.pth`, outputs lourds et archives de données restent hors Git ;
- si Streamlit échoue au chargement du modèle, vérifier d'abord le chemin du checkpoint.

Créer du data leakage :

- ne jamais entraîner sur un `pair_id` présent dans `data/processed/splits_full/val_pairs.csv` ou `test_pairs.csv` pour les protocoles no-leak ;
- ne jamais augmenter validation/test ;
- ne pas changer le test commun pour comparer deux modèles.

Poller trop souvent Rorqual :

- ne pas utiliser de boucle `watch squeue` ;
- les scripts SLURM incluent des notifications courriel ;
- commandes ponctuelles raisonnables : `squeue -u $USER`, puis consultation des logs.

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
bash slurm/submit_long250_aug_sampler_campaign.sh
bash slurm/submit_building100_sweep_v1.sh
```

La campagne damage augmentation/sampler de Jalon 3 compte 32 runs terminés. Les campagnes suivantes servent à comparer les meilleurs candidats à 250 epochs et à chercher un meilleur segmentateur bâtiment. Les jobs longs écrivent des CSV de résumé dans `outputs/predictions/`.

Vérifier un job sans polling agressif :

```bash
squeue -u $USER
tail -f ~/scratch/CrisisMap-AI/logs/<fichier>.out
```

Annuler un job :

```bash
scancel <jobid>
```

Récupérer les résultats depuis Windows se fait avec `scp`, en copiant surtout les métriques JSON/CSV, figures utiles et checkpoints choisis.
