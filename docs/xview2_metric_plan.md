# Plan de métriques xView2-style

## Objectif

Ce dossier ajoute une couche d'export et d'évaluation inspirée du format de scoring xView2. Le but est de produire des masques PNG simples, lisibles par des scripts externes, et de calculer un score pondéré proche de la logique xView2 :

```text
score = 0.3 * localization_f1 + 0.7 * damage_f1
```

Cette étape ne modifie pas l'entraînement, les checkpoints, les données ou les résultats existants.

## Scripts

### `scripts/export_xview2_format.py`

Exporte les prédictions et les cibles au format PNG `uint8` `1024x1024`.

Noms de fichiers produits :

```text
test_localization_XXXXX_prediction.png
test_damage_XXXXX_prediction.png
test_localization_XXXXX_target.png
test_damage_XXXXX_target.png
```

Le script écrit aussi :

- `manifest.csv` : correspondance entre index exporté et `pair_id`;
- `metadata.json` : checkpoint, split, mode cible et mapping de classes.

Exemple :

```bash
python scripts/export_xview2_format.py \
  --checkpoint outputs/checkpoints/unet_1024_long250_noleak_match_hist_all_aug-safe_sampler-damage-sqrt-alpha4_250epochs/best_unet.pt \
  --root data/raw/xbd/train \
  --split-csv data/processed/splits_full/test_pairs.csv \
  --output-dir outputs/predictions/xview2_style_unet_champion_d4 \
  --image-size 1024 \
  --batch-size 2 \
  --target-mode 3-class \
  --device cuda \
  --amp \
  --tta-mode d4
```

### `scripts/evaluate_xview2_style_metrics.py`

Relit les PNGs exportés et calcule :

- `localization_f1`;
- `damage_f1`;
- score pondéré.

Exemple :

```bash
python scripts/evaluate_xview2_style_metrics.py \
  --input-dir outputs/predictions/xview2_style_unet_champion_d4 \
  --target-mode 3-class \
  --output-json outputs/predictions/xview2_style_unet_champion_d4_metrics.json \
  --output-csv outputs/predictions/xview2_style_unet_champion_d4_metrics.csv
```

## Mapping des classes

### Mode actuel 3 classes

Le projet CrisisMap AI utilise actuellement :

```text
0 = background
1 = no_damage
2 = damaged
```

Le masque de localisation est dérivé ainsi :

```text
localization = damage_mask > 0
```

Le score pondéré est nommé :

```text
binary_damage_xview2_like_score
```

Ce nom est volontaire. Le score 3 classes n'est pas comparable au score officiel xView2, car les classes `minor`, `major` et `destroyed` ont été fusionnées en une seule classe `damaged`.

### Mode futur 5 classes

Le format futur visé est :

```text
0 = background
1 = no_damage
2 = minor
3 = major
4 = destroyed
```

Dans ce cas, le script calcule un :

```text
official_style_weighted_score
```

Ce score reste mask-based. Il n'est pas le script officiel polygon-level xView2, mais il suit la même idée de pondération localisation/dommages.

## Définition des métriques

### Localization F1

Le F1 de localisation est calculé sur un masque binaire bâtiment/fond :

```text
building = mask > 0
```

### Damage F1

En mode `3-class`, `damage_f1` est le F1 de la classe `damaged` (`2`).

En mode `5-class`, `damage_f1` est la moyenne macro des F1 des classes :

```text
1 = no_damage
2 = minor
3 = major
4 = destroyed
```

## Utilité

Cette couche sert à :

- comparer les modèles avec une métrique plus proche de xView2;
- exporter des masques standardisés;
- préparer le retour futur vers une segmentation multi-niveaux;
- documenter clairement la différence entre notre formulation 3 classes actuelle et le protocole officiel xView2.

## Limite importante

Les scores produits en mode `3-class` sont utiles pour le suivi interne du projet, mais ils ne doivent pas être présentés comme des scores officiels xView2.
