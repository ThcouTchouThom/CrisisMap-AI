# Plan attendu pour la présentation jalon 2

Durée cible : 6 à 8 minutes.

## 1. Contexte et objectif

Présenter le problème : après une catastrophe, il faut identifier rapidement les zones et bâtiments endommagés.

Présenter la solution : segmentation automatique sur images satellites avant/après.

## 2. Dataset xBD/xView2

Montrer :

- images avant/après ;
- masques ;
- formulation 3 classes ;
- volume brut : 2799 paires.

Mentionner `xview_geotransforms.json` comme piste future pour la cartographie géographique.

## 3. Pipeline de données

Expliquer :

- inspection ;
- index CSV ;
- visualisation ;
- splits train/validation/test ;
- filtre `min_nonzero_ratio >= 0.01`.

## 4. Exploration et difficultés

Insister sur :

- déséquilibre des classes ;
- rareté de `damaged` ;
- limites de l'accuracy ;
- intérêt de IoU damaged et F1 damaged.

## 5. Baseline U-Net

Présenter :

- entrée 6 canaux ;
- sortie 3 classes ;
- U-Net léger ;
- CE-Dice avec poids `[0.05, 1.0, 4.0]`.

## 6. Premiers résultats

Présenter le baseline 512 old4, puis mentionner les premiers résultats no-leak comme progression.

Ne pas présenter les résultats avancés comme définitifs.

## 7. Architecture technique

Montrer l'organisation :

- `src/crisismap/data` ;
- `src/crisismap/models` ;
- `src/crisismap/training` ;
- `src/crisismap/evaluation` ;
- `app/streamlit_app.py` ;
- `slurm/`.

## 8. Prochaines étapes

Mentionner :

- finaliser augmentation/sampler ;
- mettre à jour Streamlit avec le meilleur modèle ;
- analyser les erreurs ;
- tester des architectures plus fortes ;
- exploiter les geotransforms pour une carte GIS.

