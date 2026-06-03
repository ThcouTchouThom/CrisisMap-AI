# Building Segmentation v2 — plan de campagne

## Contexte

Building100 est terminé et a identifié plusieurs bons modèles de segmentation bâtiment. La campagne Building long v1 est soumise sur Rorqual, mais ses résultats ne sont pas encore consolidés à cause de la maintenance. En parallèle, les scripts de post-traitement bâtiment, TTA et ensemble existent, mais n'ont pas encore été exploités systématiquement.

Building v2 prépare la prochaine vague building-only sans modifier les sorties Building100 ni Building long v1.

## Objectif

Tester ce que Building100 n'explorait pas directement:

- entraînement par crops 512 et 608;
- crop rare-building centré sur des pixels bâtiment;
- pertes boundary-aware;
- variantes orientées rappel;
- préparation à TTA, ensemble et post-processing.

La tâche reste strictement building-only:

- 0 = fond;
- 1 = bâtiment;
- cible = `original_target > 0`.

## Modifications du training

`scripts/train_building_segmentation.py` est étendu de façon rétrocompatible:

- `--train-mode full1024|crop512|crop608`;
- `--rare-building-crop-prob`;
- `--rare-building-crop-alpha`;
- `--loss bce-dice-boundary`;
- `--loss focal-tversky-boundary`;
- métriques de validation supplémentaires de contour: `boundary_precision`, `boundary_recall`, `boundary_f1`.

Le comportement par défaut reste `full1024`, donc les anciens scripts gardent leur logique.

## Sweep v2

La config `configs/building_v2_sweep.csv` contient 15 runs de 100 epochs.

### A. Crop vs full

Compare full 1024, crop 512 et crop 608 sur les meilleurs modèles Building100:

- U-Net++ EfficientNet-B4;
- U-Net++ EfficientNet-B3;
- DeepLabV3+ ResNet50.

### B. Boundary loss

Teste si une perte de contour améliore les bâtiments fins et les limites:

- `focal-tversky-boundary`;
- poids boundary par défaut: 0.2.

### C. Recall-oriented

Teste un sampler plus agressif et des pertes alternatives:

- building-sqrt alpha 16;
- focal-dice;
- bce-dice.

### D. Split robustness

Teste si le meilleur modèle reste robuste sur:

- `splits_noleak_building_rich_002`;
- `splits_noleak_dmg001_v2`;
- `splits_noleak_match_hist_all`.

## Rorqual

Les scripts dédiés sont:

- `slurm/run_building_v2_config.sh`;
- `slurm/submit_building_v2_sweep.sh`;
- `slurm/smoke_building_v2.sbatch`.

Ils utilisent:

- `${SCRATCH}/CrisisMap-AI` pour logs, cache Triton et run logs;
- aucune partition explicite;
- notifications email;
- logique safe skip/evaluate/resume/force.

DeepLabV3+ active automatiquement `--drop-last-train` pour éviter l'erreur BatchNorm sur le dernier batch de taille 1.

## Évaluation

Chaque job complet est évalué avec les seuils:

```text
0.3, 0.4, 0.5, 0.6, 0.7
```

Le summary `outputs/predictions/building_v2_sweep_summary.csv` classe les runs par:

1. meilleur Building IoU;
2. meilleur Building F1;
3. meilleur Building recall.

Les métriques object-level issues de `evaluate_building_segmentation.py` sont conservées dans les JSON/CSV de métriques.

## Lancement

Smoke test recommandé:

```bash
sbatch slurm/smoke_building_v2.sbatch
```

Soumission de la campagne:

```bash
bash slurm/submit_building_v2_sweep.sh
```

La campagne n'a pas besoin d'être lancée tant que Building long v1 et les expériences de post-processing/ensemble ne sont pas consolidées.

## Critères de décision

Un modèle Building v2 devient candidat d'intégration damage si:

- Building IoU augmente clairement;
- Building F1 reste élevé;
- le recall bâtiment progresse sans effondrer la précision;
- les contours sont meilleurs visuellement et via `boundary_f1`;
- le downstream damage avec masque bâtiment prédit s'améliore réellement.
