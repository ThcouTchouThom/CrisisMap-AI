# Campagne Building Post-Processing v1

## Objectif

Cette campagne évalue l'effet des masques de bâtiments prédits sur la segmentation des dégâts. Elle ne lance aucun entraînement. Elle compare :

- les modèles de segmentation bâtiment seuls;
- les ensembles de modèles bâtiment;
- le post-traitement du modèle damage avec masque bâtiment prédit;
- les modes oracle bâtiment, uniquement comme borne supérieure méthodologique.

Le modèle damage de référence reste :

`outputs/checkpoints/unet_1024_long250_noleak_match_hist_all_aug-safe_sampler-damage-sqrt-alpha4_250epochs/best_unet.pt`

L'inférence damage utilise `d4` TTA, car c'est actuellement le meilleur mode d'inférence connu pour le champion U-Net.

## Modèles bâtiment testés

La campagne cible les checkpoints suivants :

- `b100_d_full_pre_unetplusplus_effb4_sampler8_focal_tversky`
- `b100_f_building_rich_002_pre_unetplusplus_effb3_focal_tversky`
- `b100_d_full_pre_deeplabv3plus_resnet50_sampler8_focal_tversky`
- `b100_a_full_pre_unet_bce_dice`

Le modèle EfficientNet-B4 est le meilleur modèle bâtiment équilibré connu. Le modèle DeepLabV3+ ResNet50 est conservé pour son rappel élevé, et le U-Net bâtiment sert de contrôle simple.

## Scripts créés

- `scripts/evaluate_building_tta_ensemble.py` : évalue un ou plusieurs modèles bâtiment, avec TTA et ensembles.
- `scripts/evaluate_downstream_building_ensemble.py` : évalue le modèle damage après post-traitement par masque bâtiment prédit ou oracle.
- `configs/building_postprocess_sweep_v1.csv` : configuration compacte des évaluations.
- `slurm/run_building_postprocess_config.sh` : exécute une ligne du CSV.
- `slurm/submit_building_postprocess_sweep_v1.sh` : soumet toutes les lignes indépendamment.

## Modes bâtiment

Les seuils testés sont :

`0.3, 0.4, 0.5, 0.6, 0.7`

Les modes TTA bâtiment sont :

- `none`
- `flips`
- `rot90`
- `d4`

Les ensembles utilisent les probabilités ou les masques binaires :

- `average_prob`
- `union`
- `intersection`
- `majority`

## Modes downstream

Les modes de post-traitement downstream sont :

- `predicted_building_clip` : force le fond hors du masque bâtiment prédit.
- `predicted_building_component_majority` : impose une décision intact/endommagé par composante bâtiment prédite.
- `oracle_building_clip` : même logique avec le masque bâtiment vérité terrain, seulement pour analyse.
- `oracle_building_component_majority` : borne supérieure par composante bâtiment vérité terrain.

Les modes oracle ne sont pas des résultats de production. Ils servent à mesurer ce que l'on pourrait gagner avec une localisation bâtiment parfaite.

## Métriques

Pour les modèles bâtiment :

- pixel accuracy;
- building IoU;
- precision bâtiment;
- rappel bâtiment;
- F1 bâtiment;
- object precision;
- object recall.

Pour le downstream damage :

- pixel accuracy;
- mean IoU;
- IoU background;
- IoU no damage;
- IoU damaged;
- precision damaged;
- recall damaged;
- F1 damaged.

Les métriques prioritaires restent `IoU damaged` et `F1 damaged`, avec une attention particulière au compromis précision/rappel.

## Utilisation Rorqual

Soumettre toute la campagne :

```bash
bash slurm/submit_building_postprocess_sweep_v1.sh
```

Soumettre une campagne avec un autre CSV :

```bash
bash slurm/submit_building_postprocess_sweep_v1.sh configs/building_postprocess_sweep_v1.csv
```

Chaque job est indépendant. Les scripts utilisent les chemins runtime sous :

`${SCRATCH}/CrisisMap-AI`

Les résultats compacts sont écrits sous :

- `outputs/predictions/building_postprocess/`
- `outputs/figures/building_postprocess/`

## Interprétation attendue

Un bon résultat downstream doit améliorer `F1 damaged` et `IoU damaged` sans sacrifier excessivement le rappel. Si un masque bâtiment prédit augmente la précision mais baisse trop le rappel, il reste utile comme piste de recherche, mais ne doit pas remplacer le modèle damage brut dans la démonstration.

La comparaison avec les modes oracle permet de distinguer deux cas :

- gain oracle fort, gain prédit faible : le concept est bon, mais le segmentateur bâtiment doit encore progresser;
- gain oracle faible : le post-traitement bâtiment n'est probablement pas le levier principal pour ce modèle damage.
