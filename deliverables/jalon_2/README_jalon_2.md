# Jalon 2 - CrisisMap AI / Aftermath

Ce dossier rassemble un paquet de livraison propre pour le jalon 2 du cours 8INF934. Il ne remplace pas le dépôt complet : il sélectionne les éléments pertinents pour démontrer que les données sont accessibles, inspectées, visualisées, séparées en ensembles train/validation/test, et qu'un premier modèle d'IA avec métriques existe.

## Objectif du projet

CrisisMap AI, aussi présenté sous le nom Aftermath, vise à produire une carte visuelle des dommages après catastrophe à partir de paires d'images satellites xBD/xView2 :

- image RGB avant catastrophe ;
- image RGB après catastrophe ;
- masque de segmentation des bâtiments et des dommages.

Pour le jalon 2, l'objectif est formulé comme une tâche de segmentation sémantique à 3 classes :

| Classe | Signification |
| --- | --- |
| 0 | arrière-plan / absence de bâtiment |
| 1 | bâtiment non endommagé |
| 2 | bâtiment endommagé |

Cette formulation est volontairement simplifiée. Le dataset original distingue plusieurs niveaux de dommages ; une étape future consistera à revenir vers une segmentation plus fine des niveaux de dommages.

## Contenu du dossier

```text
deliverables/jalon_2/
  README_jalon_2.md
  notebooklm_sources/          # Sources Markdown prêtes à importer dans NotebookLM.
  figures/                     # Petites figures copiées ou références vers figures lourdes.
  results/                     # Synthèse des résultats utiles au jalon.
  slides_draft/                # Plan de présentation 6 à 8 minutes.
```

## Données

Le dataset utilisé est xBD/xView2. Les archives ne sont pas suivies dans Git, car elles sont volumineuses. Elles doivent être placées localement dans :

```text
data/raw/archives/
```

Archives attendues :

- `train_images_labels_targets.tar`
- `xview_geotransforms.json.tgz`

Après extraction, la structure attendue est :

```text
data/raw/xbd/train/images/
data/raw/xbd/train/labels/
data/raw/xbd/train/targets/
data/raw/geotransforms/xview_geotransforms.json
```

Le fichier `xview_geotransforms.json` contient des métadonnées de géoréférencement. Il est extrait, mais il n'est pas encore utilisé par le pipeline d'entraînement. Il pourra servir plus tard à replacer les prédictions dans un contexte GIS ou cartographique.

## Scripts principaux

Inspection et indexation :

- `src/crisismap/data/inspect_xbd.py`
- `src/crisismap/data/build_xbd_index.py`
- `src/crisismap/data/summarize_xbd_index.py`

Création de splits :

- `src/crisismap/data/create_xbd_splits.py`
- `scripts/create_full_splits.ps1`
- `scripts/create_noleak_common_eval_splits.py`
- `scripts/create_advanced_noleak_train_splits.py`

Dataset et modèle :

- `src/crisismap/data/xbd_dataset.py`
- `src/crisismap/models/unet.py`

Entraînement et évaluation :

- `src/crisismap/training/train_unet.py`
- `src/crisismap/evaluation/evaluate_unet.py`
- `src/crisismap/evaluation/predict_unet_sample.py`

Prototype :

- `app/streamlit_app.py`

## Reproduction minimale

Depuis la racine du dépôt, après avoir placé les archives xBD :

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_project.ps1
```

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
  --output-dir outputs/checkpoints/unet_baseline_512_v2_30epochs `
  --image-size 512 `
  --batch-size 2 `
  --epochs 30 `
  --target-mode 3-class
```

Évaluation :

```powershell
python src/crisismap/evaluation/evaluate_unet.py `
  --root data/raw/xbd/train `
  --split-csv data/processed/splits/test_pairs.csv `
  --checkpoint outputs/checkpoints/unet_baseline_512_v2_30epochs/best_unet.pt `
  --output outputs/predictions/unet_512_v2_30epochs_test_metrics.json `
  --image-size 512 `
  --target-mode 3-class
```

Prototype Streamlit :

```powershell
streamlit run app/streamlit_app.py
```

## Résultats à présenter

Le baseline local 512 sur le split initial old4 a permis de valider la chaîne complète :

| Modèle | Mean IoU | IoU damaged | F1 damaged | Pixel accuracy |
| --- | ---: | ---: | ---: | ---: |
| U-Net 512 old4, 30 epochs | 0.6257 | 0.3870 | 0.5581 | 0.9175 |

Les expériences propres no-leak plus récentes dépassent le strict besoin du jalon 2, mais montrent la maturité du projet :

| Protocole | Mean IoU | IoU damaged | F1 damaged |
| --- | ---: | ---: | ---: |
| 1024 no-leak, match_hist1000, 100 epochs | ~0.6578 | ~0.4159 | ~0.5874 |
| 1024 no-leak, match_hist1000, 250 epochs | ~0.6651 | ~0.4175 | ~0.5891 |

Ces scores doivent être présentés comme des résultats en cours d'amélioration, non comme une performance finale.

## Ce qui est complet

- Chargement et inspection du dataset xBD/xView2.
- Visualisation des images avant/après et des masques.
- Index CSV des paires image/label/masque.
- Splits train/validation/test.
- Baseline U-Net fonctionnel.
- Entraînement local et sur Rorqual.
- Métriques quantitatives : accuracy, IoU, F1, précision, rappel.
- Prototype Streamlit fonctionnel.
- Correction méthodologique du data leakage avec protocole no-leak.

## Ce qui est planifié

- Finaliser les expériences augmentation/sampler.
- Mettre à jour le prototype avec le meilleur modèle no-leak final.
- Ajouter une analyse qualitative des erreurs.
- Explorer des architectures plus fortes : Siamese U-Net, SegFormer, ChangeFormer ou modèles hybrides segmentation/classification.
- Exploiter `xview_geotransforms.json` pour une vraie visualisation cartographique.

