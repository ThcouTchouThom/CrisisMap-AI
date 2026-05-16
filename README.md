# Aftermath

**Voir les dégâts pour agir plus vite.**

Aftermath, aussi appelé CrisisMap AI dans le code, est un prototype académique de segmentation des dommages aux bâtiments après catastrophe. Le projet utilise des paires d'images satellites avant/après du dataset xBD/xView2 afin de produire des cartes de dommages au niveau pixel.

Équipe : Thomas Gourjault, Aurélien Casagrandi, Grégory Jourdain.

## Objectif

Le but est de construire une chaîne complète :

```text
archives xBD/xView2 -> extraction -> index CSV -> splits -> Dataset PyTorch -> U-Net -> métriques -> prototype
```

Le prototype actuel répond à une formulation de segmentation sémantique à 3 classes :

| Classe | Signification |
| --- | --- |
| `0` | background / absence de bâtiment |
| `1` | bâtiment non endommagé |
| `2` | bâtiment endommagé |

L'entrée du modèle est un tenseur à 6 canaux : image RGB pré-catastrophe + image RGB post-catastrophe. La segmentation multi-niveaux des dommages, plus proche du format original xBD, est un objectif futur.

## Dataset

Le projet utilise le jeu d'entraînement xBD/xView2. Les données brutes, images extraites, checkpoints et sorties générées ne sont pas versionnés dans Git.

Archives attendues localement :

```text
data/raw/archives/train_images_labels_targets.tar
data/raw/archives/xview_geotransforms.json.tgz
```

Structure attendue après extraction :

```text
data/raw/xbd/train/images/
data/raw/xbd/train/labels/
data/raw/xbd/train/targets/
data/raw/geotransforms/xview_geotransforms.json
```

`xview_geotransforms.json` contient des métadonnées de géoréférencement utiles pour replacer plus tard les prédictions sur une carte ou dans un outil GIS. Il est extrait, mais pas encore utilisé dans l'entraînement.

## Structure du dépôt

```text
app/                         # Prototype Streamlit.
configs/                     # Configuration éventuelle des expériences.
data/                        # Données locales et CSV traités; les données lourdes ne sont pas suivies.
deliverables/                # Livrables de cours, dont le jalon 2.
notebooks/                   # Exploration.
outputs/                     # Checkpoints, figures, métriques; non suivis.
scripts/                     # Scripts utilitaires locaux et génération de splits.
slurm/                       # Scripts Alliance / Rorqual.
src/crisismap/
  data/                      # Inspection, indexation, splits, Dataset PyTorch.
  evaluation/                # Évaluation et visualisation de prédictions.
  models/                    # U-Net.
  training/                  # Entraînement.
  visualization/             # Figures dataset et métriques.
```

## Pipeline de données

Les scripts principaux sont :

- `src/crisismap/data/inspect_xbd.py` : inspection de la structure xBD.
- `src/crisismap/data/build_xbd_index.py` : construction de `data/processed/xbd_train_index.csv`.
- `src/crisismap/data/summarize_xbd_index.py` : statistiques exploratoires.
- `src/crisismap/data/create_xbd_splits.py` : splits simples train/validation/test.
- `scripts/create_advanced_noleak_train_splits.py` : splits avancés sans fuite de données.
- `src/crisismap/data/xbd_dataset.py` : Dataset PyTorch 6 canaux.

Le dataset brut contient 2799 paires pré/post. Certains splits filtrent les images avec trop peu d'information bâtiment, notamment avec `min_nonzero_ratio >= 0.01`.

## Méthodologie actuelle

Les premières comparaisons ont révélé un risque de data leakage entre certains splits d'entraînement et un test global. Le protocole actuel utilise donc une validation et un test communs :

```text
common_val  = data/processed/splits_full/val_pairs.csv
common_test = data/processed/splits_full/test_pairs.csv
```

Les nouveaux splits d'entraînement excluent tous les `pair_id` présents dans ces deux fichiers. La validation et le test ne sont pas augmentés et ne sont pas pondérés par sampler.

## Baseline IA

Le baseline est un U-Net léger :

- entrée : 6 canaux, pré RGB + post RGB ;
- sortie : 3 classes ;
- perte de référence : Cross-Entropy pondérée + Dice loss ;
- poids de référence : `[0.05, 1.0, 4.0]` ;
- expériences réalisées en 512 et 1024 pixels.

Le baseline local 512 a validé toute la chaîne. Les expériences 1024 no-leak sur Rorqual servent à améliorer la classe `damaged`, qui reste la plus difficile et la plus importante.

## Installation locale

Créer un environnement Python puis installer les dépendances :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Pour une installation complète à partir des archives locales, utiliser le script PowerShell :

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_project.ps1
```

Pour reconstruire les fichiers traités :

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_project.ps1 -Force
```

## Commandes utiles

Inspection :

```powershell
python src/crisismap/data/inspect_xbd.py --root data/raw/xbd/train
```

Visualisation d'un échantillon :

```powershell
python src/crisismap/visualization/visualize_xbd_sample.py --root data/raw/xbd/train --mode 3-class
```

Indexation :

```powershell
python src/crisismap/data/build_xbd_index.py --root data/raw/xbd/train --output data/processed/xbd_train_index.csv
```

Entraînement baseline :

```powershell
python src/crisismap/training/train_unet.py `
  --root data/raw/xbd/train `
  --train-csv data/processed/splits/train_pairs.csv `
  --val-csv data/processed/splits/val_pairs.csv `
  --output-dir outputs/checkpoints/unet_baseline `
  --image-size 512 `
  --batch-size 2 `
  --epochs 5 `
  --target-mode 3-class `
  --loss ce-dice `
  --class-weights 0.05 1.0 4.0
```

Évaluation :

```powershell
python src/crisismap/evaluation/evaluate_unet.py `
  --root data/raw/xbd/train `
  --split-csv data/processed/splits/test_pairs.csv `
  --checkpoint outputs/checkpoints/unet_baseline/best_unet.pt `
  --output outputs/predictions/unet_baseline_test_metrics.json `
  --image-size 512 `
  --target-mode 3-class
```

Visualisation d'une prédiction :

```powershell
python src/crisismap/evaluation/predict_unet_sample.py `
  --root data/raw/xbd/train `
  --split-csv data/processed/splits/test_pairs.csv `
  --checkpoint outputs/checkpoints/unet_baseline/best_unet.pt `
  --image-size 512 `
  --target-mode 3-class
```

## Prototype Streamlit

L'application Streamlit permet de sélectionner un split et une paire d'images, puis d'afficher l'image avant, l'image après, le masque vérité terrain, la prédiction et une superposition sur l'image post-catastrophe.

```powershell
streamlit run app/streamlit_app.py
```

Le checkpoint utilisé par défaut doit être placé dans `outputs/checkpoints/`. Les checkpoints ne sont pas inclus dans Git.

## Rorqual / SLURM

Les entraînements lourds sont préparés pour Alliance / Calcul Québec, notamment Rorqual H100. Les fichiers utiles sont dans `slurm/` :

- `slurm/setup_rorqual.sh` : installation côté cluster et préparation des données.
- `slurm/smoke_unet_512.sbatch` : test technique court.
- `slurm/train_unet_full_1024.sbatch` : entraînement 1024 complet.
- `slurm/sweep_*.sbatch` : campagnes de splits, augmentation et samplers.

Les scripts utilisent les répertoires `~/work/CrisisMap-AI` pour le code et `~/scratch/CrisisMap-AI` pour les données, sorties et logs. Ils incluent des notifications courriel pour éviter un polling fréquent du scheduler.

## Livrables

Le dossier du jalon 2 est ici :

```text
deliverables/jalon_2/README_jalon_2.md
```

Il contient une synthèse en français, des sources NotebookLM, un plan de présentation, des résultats résumés et quelques petites figures.

## État actuel

Terminé :

- pipeline de données xBD/xView2 ;
- visualisations ;
- Dataset PyTorch ;
- U-Net baseline ;
- métriques d'évaluation ;
- protocole no-leak ;
- scripts SLURM ;
- prototype Streamlit.

En cours :

- campagne augmentation/sampler train-only ;
- comparaison des meilleurs splits no-leak ;
- mise à jour du prototype avec le meilleur checkpoint final.

Étapes futures :

- analyse qualitative des erreurs ;
- segmentation multi-niveaux des dommages ;
- exploitation des geotransforms pour une visualisation cartographique ;
- architectures plus fortes : Siamese U-Net, SegFormer, ChangeFormer, modèles hybrides segmentation/classification.
