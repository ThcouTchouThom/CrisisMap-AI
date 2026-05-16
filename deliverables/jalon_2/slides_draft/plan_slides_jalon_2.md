# Plan de présentation - Jalon 2

Durée cible : 6 à 8 minutes.

## Slide 1 - Contexte et objectif

- Catastrophes naturelles : besoin d'estimer rapidement les dommages.
- Objectif : cartographie automatique des bâtiments endommagés.
- Projet : CrisisMap AI / Aftermath.

Message oral : nous construisons un prototype qui transforme une paire satellite avant/après en carte de dommages.

## Slide 2 - Dataset xBD/xView2

- Images satellites pré-catastrophe et post-catastrophe.
- Annotations bâtiments et dommages.
- Masques de segmentation.
- 2799 paires brutes.
- Formulation actuelle à 3 classes.

Visuel suggéré : `outputs/figures/first_xbd_sample.png`.

## Slide 3 - Pipeline de données

- Extraction locale des archives.
- Inspection automatique.
- Index CSV.
- Visualisation.
- Splits train/validation/test.
- Filtre `min_nonzero_ratio >= 0.01`.

Mentionner que les données brutes et checkpoints ne sont pas dans Git.

## Slide 4 - Exploration et difficultés

- Distribution par catastrophe.
- Déséquilibre fort : background dominant, damaged rare.
- Pixel accuracy insuffisante.
- Métriques importantes : mean IoU, IoU damaged, F1 damaged.

Visuel suggéré : masque ou exemple 3 classes.

## Slide 5 - Baseline U-Net

- Entrée : pré RGB + post RGB = 6 canaux.
- Sortie : 3 classes.
- U-Net léger.
- CE-Dice loss.
- Poids `[0.05, 1.0, 4.0]`.
- Expériences 512 puis 1024.

## Slide 6 - Premiers résultats

Table courte :

| Modèle | Mean IoU | IoU damaged | F1 damaged |
| --- | ---: | ---: | ---: |
| U-Net 512 old4 | 0.6257 | 0.3870 | 0.5581 |
| U-Net 1024 no-leak avancé | ~0.658-0.665 | ~0.416-0.418 | ~0.587-0.589 |

Message : le baseline fonctionne, mais la classe damaged reste le défi central.

## Slide 7 - Architecture technique et prototype

- `src/crisismap/data` : inspection, index, splits, dataset PyTorch.
- `src/crisismap/models` : U-Net.
- `src/crisismap/training` : entraînement.
- `src/crisismap/evaluation` : métriques et prédictions.
- `app/streamlit_app.py` : prototype interactif.
- `slurm/` : préparation Rorqual H100.

Mentionner le protocole no-leak comme correction méthodologique.

## Slide 8 - Prochaines étapes

- Finaliser campagne augmentation/sampler.
- Mettre à jour Streamlit avec le meilleur modèle no-leak.
- Exploiter `xview_geotransforms.json` pour une carte GIS.
- Tester Siamese U-Net, SegFormer, ChangeFormer ou modèles hybrides.
- Faire une analyse d'erreurs : faux positifs, faux négatifs, rappel damaged.

