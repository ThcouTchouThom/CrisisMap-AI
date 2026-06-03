# xView2 Strong Baseline v2

## Objectif

La campagne v2 étend la première vague `xView2 strong baseline` avec des variantes ciblées, sans faire de recherche aléatoire. Le but est de tester les hypothèses les plus utiles avant de lancer de longs entraînements :

- meilleurs backbones;
- fusions temporelles plus expressives;
- pertes adaptées au déséquilibre;
- crops 512/608 biaisés vers les dégâts;
- formulation multi-head bâtiment/dégât;
- préparation future du mode officiel 5 classes.

La campagne v1 reste inchangée.

## Fichiers

- `configs/xview2_strong_baseline_sweep_v2.csv`
- `slurm/run_xview2_strong_baseline_v2_config.sh`
- `slurm/submit_xview2_strong_baseline_sweep_v2.sh`

Le modèle et le script d'entraînement existants ont été étendus pour supporter v2 :

- `src/crisismap/models/xview2_strong_baseline.py`
- `scripts/train_xview2_strong_baseline.py`

## Modèles supportés

Backbones :

- ResNet34;
- ResNet50;
- EfficientNet-B3 via `timm`.

Fusions :

- `shared` : `concat(pre, post, abs(post-pre))`;
- `absdiff` : `concat(post, abs(post-pre))`;
- `abs_signed` : `concat(pre, post, abs(post-pre), post-pre)`;
- `abs_signed_product` : ajoute aussi `pre * post`;
- attention channel-wise légère sur la fusion.

## Pertes

La v2 supporte :

- `ce-dice`;
- `focal-dice`;
- `focal-tversky`.

En mode `multilabel-building-damage`, `ce-dice` correspond à `BCE + Dice` sur les têtes binaires bâtiment et dommage.

## Crops rares

Les runs `rare_alpha` utilisent :

```text
rare_crop_probability = alpha / (1 + alpha)
```

Exemples :

- alpha 2 : environ 66 % de crops centrés sur des pixels endommagés si disponibles;
- alpha 4 : environ 80 %;
- alpha 8 : environ 89 %.

Ce mécanisme ne change que l'entraînement. Validation et test restent en image complète sans augmentation.

## Runs v2

La campagne contient 20 lignes :

- 18 runs activés;
- 2 runs 5 classes désactivés et planifiés.

### A. Architecture / backbone

1. `resnet34_unet_shared_1024_3class_ce_dice`
2. `resnet34_unet_shared_crop512_3class_ce_dice`
3. `resnet50_unet_shared_crop512_3class_ce_dice`
4. `efficientnet_b3_unet_shared_crop512_3class_ce_dice`

### B. Fusion

5. `resnet34_absdiff_crop512_3class`
6. `resnet34_abs_signed_crop512_3class`
7. `resnet34_abs_signed_product_crop512_3class`
8. `resnet34_attention_fusion_crop512_3class`

### C. Loss

9. `resnet34_attention_crop512_focal_dice`
10. `resnet34_attention_crop512_focal_tversky`
11. `resnet34_abs_signed_crop512_focal_dice`
12. `resnet34_abs_signed_crop512_focal_tversky`

### D. Sampling / crop

13. `resnet34_attention_crop512_rare_alpha2`
14. `resnet34_attention_crop512_rare_alpha4`
15. `resnet34_attention_crop512_rare_alpha8`
16. `resnet34_attention_crop608_rare_alpha4`

### E. Multi-head / multilabel

17. `resnet34_attention_crop512_multilabel_building_damage`
18. `resnet50_attention_crop512_multilabel_building_damage`

### F. Future 5-class

19. `resnet34_attention_crop512_5class`
20. `resnet34_abs_signed_crop512_5class`

Ces deux lignes sont dans le CSV avec `enabled=0`. Elles ne sont pas soumises par défaut.

## Audit 5 classes

Le runner v2 contient un audit automatique pour les lignes `target_mode=5-class` ou `label_mode=5-class`.

Si une ligne 5 classes est activée, le runner vérifie que les labels séparés `2`, `3`, `4` sont observés dans les splits. Si les labels originaux ne contiennent pas les classes `minor`, `major`, `destroyed` séparément, le run échoue avant l'entraînement.

Cela évite de prétendre entraîner un modèle xView2 officiel alors que le dataset actif serait encore en formulation 3 classes.

## Lancement

Soumettre uniquement les lignes activées :

```bash
bash slurm/submit_xview2_strong_baseline_sweep_v2.sh
```

Soumettre avec un CSV explicite :

```bash
bash slurm/submit_xview2_strong_baseline_sweep_v2.sh configs/xview2_strong_baseline_sweep_v2.csv
```

Le submitter ignore les lignes `enabled=0`.

## Sorties

Les checkpoints sont distincts :

```text
outputs/checkpoints/<experiment>/
```

Les métriques test v2 sont écrites dans :

```text
outputs/predictions/xview2_strong_baseline_v2/
```

Les logs/cache Rorqual utilisent :

```text
${SCRATCH}/CrisisMap-AI
```

## Smoke test

Le smoke test existant a été étendu pour couvrir toutes les familles supportées, y compris EfficientNet-B3 et les sorties damage à 1, 3 et 5 canaux :

```bash
sbatch slurm/smoke_xview2_strong_baseline.sbatch
```

## Critères de décision

Les métriques prioritaires restent :

- F1 damaged;
- IoU damaged;
- recall damaged;
- precision damaged;
- mean IoU.

Les meilleurs runs v2 pourront ensuite être comparés au champion U-Net + TTA d4 et aux meilleurs modèles Siamese v2.
