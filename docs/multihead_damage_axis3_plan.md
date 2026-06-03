# Axis 3 — Multi-head building localization + damage prediction

## Objectif

Axis 3 ajoute un pipeline multi-task pour apprendre simultanément:

1. une localisation binaire des bâtiments;
2. une prédiction de dommage.

Cette piste ne remplace pas les pipelines existants. Elle sert à tester si une supervision explicite de la localisation bâtiment peut améliorer la prédiction `damaged`, surtout après les résultats oracle montrant qu'un masque bâtiment parfait apporte un gain important.

## Entrées et sorties

Entrée:

- image pré-catastrophe RGB;
- image post-catastrophe RGB;
- concaténées en tenseur 6 canaux.

Sorties:

- `building_logits`: masque binaire bâtiment/fond;
- `damage_logits`: tête dommage configurable.

## Architectures

Le backbone est Siamese: les images `pre` et `post` passent dans le même encodeur avec poids partagés. Les variantes v1 sont:

- ResNet34 + attention;
- ResNet50 + attention;
- EfficientNet-B3 + attention;
- ResNet34 + fusion `abs_signed`.

La fusion principale est:

```text
concat(pre_feat, post_feat, abs(post_feat - pre_feat))
```

La variante `abs_signed` ajoute aussi `post_feat - pre_feat`.

## Modes de dommage

### Mode A — 3 classes

Formulation actuelle:

- 0 fond;
- 1 bâtiment non endommagé;
- 2 bâtiment endommagé.

La perte dommage est CE-Dice par défaut.

### Mode B — building-only 2 classes

La tête dommage prédit seulement:

- bâtiment non endommagé;
- bâtiment endommagé.

La perte dommage est masquée sur les pixels bâtiment (`target > 0`). La prédiction finale recompose un masque 3 classes avec la tête bâtiment.

### Mode C — 5 classes xView2 futur

Les lignes 5 classes sont présentes mais désactivées dans la config v1. Le runner contient un audit automatique: il vérifie la présence de labels séparés `minor`, `major`, `destroyed` avant de lancer un run 5 classes. Sans cet audit positif, ces lignes restent `planned`.

## Pertes

La perte totale est:

```text
total_loss = lambda_building * building_loss + lambda_damage * damage_loss
```

Pertes bâtiment:

- BCE-Dice;
- Focal-Tversky.

Pertes dommage:

- CE-Dice;
- Focal-Dice;
- Focal-Tversky;
- masked CE ou masked focal pour le mode building-only.

## Sweep v1

La campagne v1 contient 12 lignes:

- 10 runs actifs de 100 epochs;
- 2 runs 5 classes désactivés jusqu'à disponibilité des labels.

Tous les runs utilisent:

- split `splits_noleak_match_hist_all`;
- image size 1024;
- crop 512 sauf le contrôle full 1024;
- AdamW;
- AMP;
- augmentations géométriques sûres;
- dossiers de checkpoints distincts.

## Évaluation

Chaque job écrit:

- métriques dommage standard;
- métriques localisation bâtiment;
- métriques dommage contraintes par la tête bâtiment;
- métriques downstream `clip`, `component majority` et oracle de référence.

Pour le mode 3 classes, le script produit aussi un score `binary_damage_xview2_like_score`. Ce score n'est pas le score officiel xView2 5 classes.

## Lancement

Smoke test:

```bash
sbatch slurm/smoke_multihead_damage.sbatch
```

Campagne:

```bash
bash slurm/submit_multihead_damage_sweep_v1.sh
```

Les jobs sont indépendants, n'utilisent pas de partition explicite, et les logs/cache runtime sont placés sous `${SCRATCH}/CrisisMap-AI`.

## Critères de décision

Un modèle Axis 3 devient candidat long training si:

- il bat le champion actuel en F1 damaged ou IoU damaged;
- il améliore le rappel damaged sans effondrer la précision;
- il améliore les métriques downstream avec masque bâtiment prédit;
- il apporte une meilleure stabilité entre damage et building localization.

Les résultats seront comparés au champion actuel U-Net + TTA d4 et aux architectures Siamese attention de l'Axis 2.
