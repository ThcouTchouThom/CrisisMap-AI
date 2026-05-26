# Aftermath

**Voir les dégâts pour agir plus vite.**

Aftermath, aussi appelé CrisisMap AI dans le code, est un projet académique de segmentation des dommages aux bâtiments après catastrophe. Il utilise des paires d'images satellites avant/après issues du dataset xBD/xView2 afin de produire une carte visuelle des bâtiments intacts et endommagés.

Équipe : Thomas GOURJAULT, Grégory JOURDAIN, Aurélien CASAGRANDI, Matthis LAHARGOUE.

## Objectif

Le projet construit une chaîne complète :

```text
archives xBD/xView2 -> extraction -> index CSV -> splits -> Dataset PyTorch -> modèle -> métriques -> prototype Streamlit
```

La formulation IA actuelle est une segmentation sémantique à 3 classes :

| Classe | Signification |
| --- | --- |
| `0` | fond / absence de bâtiment |
| `1` | bâtiment non endommagé |
| `2` | bâtiment endommagé |

L'entrée du modèle damage est un tenseur à 6 canaux : image RGB pré-catastrophe + image RGB post-catastrophe. La segmentation multi-niveaux des dommages, plus proche du format original xBD, reste un objectif futur.

## Dataset

Le projet utilise le jeu d'entraînement xBD/xView2. Les données brutes, les images extraites, les checkpoints et les sorties générées ne sont pas versionnés dans Git.

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

`xview_geotransforms.json` contient des métadonnées de géoréférencement utiles pour replacer plus tard les prédictions dans une carte ou un outil GIS. Ces données sont extraites, mais elles ne sont pas encore intégrées à l'entraînement.

## Structure du dépôt

```text
app/                         # Prototype Streamlit.
configs/                     # Notes et configurations éventuelles.
data/                        # Données locales et CSV traités; les fichiers lourds sont ignorés.
deliverables/                # Sources légères de livrables de cours.
docs/                        # Documentation projet et onboarding.
outputs/                     # Checkpoints, figures, métriques; non suivis.
scripts/                     # Scripts utilitaires, splits avancés, évaluations complémentaires.
slurm/                       # Scripts Alliance / Rorqual.
src/crisismap/
  data/                      # Inspection, indexation, splits, Dataset PyTorch.
  evaluation/                # Évaluation et visualisation de prédictions.
  models/                    # U-Net.
  training/                  # Entraînement damage.
  visualization/             # Figures dataset et métriques.
```

## Pipeline de données

Les scripts principaux sont :

- `src/crisismap/data/inspect_xbd.py` : inspection de la structure xBD.
- `src/crisismap/data/build_xbd_index.py` : construction de `data/processed/xbd_train_index.csv`.
- `src/crisismap/data/summarize_xbd_index.py` : statistiques exploratoires.
- `src/crisismap/data/create_xbd_splits.py` : création de splits simples.
- `scripts/create_noleak_common_eval_splits.py` : création de splits sans fuite autour d'une validation et d'un test communs.
- `scripts/create_advanced_noleak_train_splits.py` : génération de splits no-leak avancés.
- `src/crisismap/data/xbd_dataset.py` : Dataset PyTorch pour les paires pré/post.

Le dataset brut contient 2 799 paires pré/post. Certains splits filtrent les images avec trop peu d'information bâtiment, notamment via `min_nonzero_ratio >= 0.01`.

## Méthodologie no-leak

Des comparaisons intermédiaires ont révélé un risque de fuite de données entre certains splits d'entraînement et un test global. Le protocole actuel fixe donc une validation et un test communs :

```text
common_val  = data/processed/splits_full/val_pairs.csv
common_test = data/processed/splits_full/test_pairs.csv
```

Les nouveaux splits d'entraînement excluent tous les `pair_id` présents dans ces deux fichiers. La validation et le test ne sont jamais augmentés et n'utilisent pas de sampler pondéré.

## Baseline damage

Le baseline principal est un U-Net :

- entrée : 6 canaux, pré RGB + post RGB ;
- sortie : 3 classes ;
- perte de référence : Cross-Entropy pondérée + Dice loss ;
- poids de référence : `[0.05, 1.0, 4.0]` ;
- expériences en 512 et 1024 pixels.

La référence no-leak actuelle donne environ `IoU damaged = 0.4175` et `F1 damaged = 0.5891`. La classe endommagée reste la plus difficile, car elle est rare et moins régulière que le fond ou les bâtiments intacts.

## Oracle et branche bâtiment

Une expérience oracle a mesuré le gain théorique d'une segmentation bâtiment parfaite :

- prédiction brute : `IoU damaged ≈ 0.4175`, `F1 damaged ≈ 0.5891` ;
- oracle building clip : `IoU damaged ≈ 0.4782`, `F1 damaged ≈ 0.6470` ;
- oracle component majority : `IoU damaged ≈ 0.5383`, `F1 damaged ≈ 0.6999`.

Ce résultat motive une future architecture en deux étapes : segmenter les bâtiments, puis classifier les dommages par pixel ou par composante bâtiment.

Une branche building-only a été ajoutée pour la tâche binaire `fond / bâtiment`, avec `target = original_target > 0`. Le premier modèle testé est un U-Net++ EfficientNet-B3 en entrée pré-catastrophe, avec focal Tversky. Les métriques préliminaires de validation sont environ `building IoU = 0.6536`, `recall = 0.8120`, `F1 = 0.7905`.

## Prototype Streamlit

Le prototype alpha Streamlit permet :

- de sélectionner une paire du dataset xBD ;
- de téléverser une paire réelle pré/post ;
- de lancer l'inférence ;
- d'afficher en priorité la superposition de la prédiction sur l'image post-catastrophe ;
- d'afficher les détails : image avant, image après, masque prédit ;
- d'afficher des métriques seulement en mode dataset, quand la vérité terrain existe.

Commande :

```powershell
streamlit run app/streamlit_app.py
```

Le checkpoint de démonstration attendu n'est pas dans Git :

```text
outputs/checkpoints/unet_1024_ce_dice_w005_1_4_noleak_match_hist1000_bs2_250epochs/best_unet_portable.pt
```

## Installation locale

Créer un environnement Python puis installer les dépendances :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Installation complète à partir des archives locales :

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_project.ps1
```

Reconstruction forcée des fichiers traités :

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_project.ps1 -Force
```

## Commandes utiles

Inspection :

```powershell
python src/crisismap/data/inspect_xbd.py --root data/raw/xbd/train
```

Indexation :

```powershell
python src/crisismap/data/build_xbd_index.py --root data/raw/xbd/train --output data/processed/xbd_train_index.csv
```

Entraînement damage :

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

Oracle building mask :

```powershell
python scripts/evaluate_oracle_building_mask_gain.py `
  --root data/raw/xbd/train `
  --split-csv data/processed/splits_full/test_pairs.csv `
  --checkpoint outputs/checkpoints/unet_1024_ce_dice_w005_1_4_noleak_match_hist1000_bs2_250epochs/best_unet.pt `
  --output-json outputs/predictions/oracle_building_mask_metrics.json
```

Segmentation bâtiment :

```powershell
python scripts/train_building_segmentation.py `
  --root data/raw/xbd/train `
  --train-csv data/processed/splits_noleak_full_train/train_pairs.csv `
  --val-csv data/processed/splits_noleak_full_train/val_pairs.csv `
  --output-dir outputs/checkpoints/building_pre_unetplusplus_effb3 `
  --model unetplusplus_effb3 `
  --input-mode pre `
  --target-mode building-binary
```

## Rorqual / SLURM

Les entraînements lourds sont préparés pour Alliance / Calcul Québec, notamment Rorqual H100. Les fichiers utiles sont dans `slurm/`.

Repères :

- code : `~/work/CrisisMap-AI` ;
- données et sorties : `~/scratch/CrisisMap-AI` ;
- venv : `~/virtualenvs/crisismap-ai` ;
- modules : `StdEnv/2023`, `python/3.11`, `gcc`, `arrow/23.0.1`, `cuda`, `opencv/4.13.0`.

Les scripts SLURM incluent des notifications courriel afin d'éviter le polling fréquent du scheduler. Il faut préférer des vérifications ponctuelles avec `squeue -u $USER` et la lecture des logs.

## Livrables

Le dossier local de rendu Jalon 3 est ignoré par Git :

```text
RENDU_JALON_3_Aftermath/
RENDU_JALON_3_Aftermath.zip
```

Il contient un rapport unique, des scripts représentatifs, des exemples légers et des notes expliquant les checkpoints exclus pour respecter les limites d'upload.

Des sources Markdown légères pour NotebookLM sont aussi disponibles dans :

```text
deliverables/jalon_3/notebooklm_sources/
```

## État actuel et prochaines étapes

Terminé :

- pipeline xBD/xView2 ;
- visualisations et index CSV ;
- Dataset PyTorch ;
- U-Net baseline ;
- protocole no-leak ;
- expérience oracle bâtiment ;
- première branche building-only ;
- prototype Streamlit alpha avec mode upload.

En cours :

- consolidation de la campagne Rorqual augmentation/sampler ;
- sélection du meilleur modèle damage ;
- évaluation test de la segmentation bâtiment.

Étapes futures :

- intégrer un masque bâtiment prédit dans le post-processing damage ;
- améliorer les contours et réduire les faux positifs ;
- revenir à une segmentation multi-niveaux des dommages ;
- exploiter les geotransforms pour une carte géoréférencée ;
- tester des architectures plus fortes : Siamese U-Net, SegFormer, ChangeFormer, modèles hybrides segmentation/classification.
