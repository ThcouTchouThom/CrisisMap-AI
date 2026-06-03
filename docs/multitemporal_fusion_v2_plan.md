# Multi-Temporal Fusion v2 — plan de campagne

## Objectif

Cette campagne prépare une deuxième vague Multi-Temporal Fusion centrée sur le dommage xBD/xView2. Elle garde l'idée principale du papier: traiter les images avant et après catastrophe avec des chemins temporels séparés, puis fusionner les cartes de caractéristiques avant la tête de segmentation.

La v1 couvrait les briques minimales: FPN, DeepLab, EfficientNet-B3, contrôle 6 canaux, contrôle Siamese attention, modes 3 classes et multilabel. La v2 est plus ciblée et plus ambitieuse, avec 23 runs de 100 epochs.

## Protocole commun

- Données: `splits_noleak_match_hist_all`.
- Taille image: 1024.
- Modes d'entraînement: crop 512, crop 608 et contrôle full 1024.
- Optimiseur: AdamW.
- AMP activé sur H100.
- Augmentations géométriques sûres.
- Échantillonnage crop rare-damage-biased pour mieux exposer la classe endommagée.
- Évaluation standard 3 classes: fond, bâtiment non endommagé, bâtiment endommagé.
- Les sorties sont écrites dans des dossiers d'expérience distincts.

## Blocs testés

### A. Backbone et décodeur

Ce bloc compare ResNet34, ResNet50, EfficientNet-B3, FPN et DeepLab. Il permet de vérifier si le gain vient surtout du backbone, du décodeur ou de la fusion temporelle.

### B. Variantes de fusion

Les variantes testées sont:

- fusion de base `pre, post, abs(post-pre)`;
- `absdiff`;
- `abs_signed`;
- `abs_signed_product`;
- attention par canaux;
- fusion gated.

L'objectif est de savoir quelle représentation du changement aide le plus la classe damaged.

### C. Pertes

Le bloc loss compare:

- CE-Dice;
- Focal-Dice;
- Focal-Tversky.

On priorise IoU damaged, F1 damaged, précision damaged, rappel damaged et mean IoU.

### D. Sampling et crop

Les runs alpha 2/4/8 testent l'intensité du crop rare-damage-biased. Le crop 608 sert de compromis entre contexte spatial et focalisation sur les zones informatives.

### E. Multi-head

Les runs multilabel apprennent une tête localisation bâtiment et une tête dommage binaire compatible avec la formulation actuelle. Ce n'est pas encore le scoring officiel xView2 5 classes.

### F. Contrôles

Deux contrôles sont conservés:

- FPN ResNet50 6 canaux;
- Siamese attention actuel en crop 512 et full 1024.

## 5 classes xView2

La v2 n'active pas de ligne 5 classes. Un audit automatique est cependant présent dans le runner: si une future ligne `5-class` est activée, les CSV et masques doivent montrer des labels séparés `minor`, `major` et `destroyed`. Sans cela, le job échoue explicitement au lieu de prétendre produire une métrique officielle xView2.

## Lancement

```bash
bash slurm/submit_multitemporal_fusion_sweep_v2.sh
```

Le submitter lit `configs/multitemporal_fusion_sweep_v2.csv` et soumet un job indépendant par ligne active. Les scripts n'utilisent pas de partition explicite et les caches/logs runtime utilisent `${SCRATCH}/CrisisMap-AI`.

## Smoke test

Avant la campagne:

```bash
sbatch slurm/smoke_multitemporal_fusion_v2.sbatch
```

Le smoke test instancie toutes les familles de modèles et vérifie un forward dummy `[1, 6, 256, 256]`.

## Critères de décision

Un modèle v2 devient candidat long training seulement s'il améliore clairement le champion actuel ou s'il montre un avantage stratégique:

- meilleur F1 damaged;
- meilleur IoU damaged;
- meilleur rappel sans chute excessive de précision;
- meilleure stabilité mean IoU;
- complémentarité possible avec les modèles Siamese attention existants.
