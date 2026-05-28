# Axe 2 - Architectures plus fortes pour la prédiction des dégâts

## Point de départ

Le champion officiel actuel reste:

```text
unet_1024_long250_noleak_match_hist_all_aug-safe_sampler-damage-sqrt-alpha4_250epochs
```

Métriques sans TTA:

| Métrique | Valeur |
|---|---:|
| Mean IoU | 0.676624 |
| IoU damaged | 0.446452 |
| Precision damaged | 0.605233 |
| Recall damaged | 0.629871 |
| F1 damaged | 0.617307 |

La meilleure inférence actuelle utilise le même modèle avec TTA `d4`:

| Métrique | Valeur |
|---|---:|
| Mean IoU | 0.681574 |
| IoU damaged | 0.461240 |
| Precision damaged | 0.635361 |
| Recall damaged | 0.627289 |
| F1 damaged | 0.631300 |

Les runs U-Net supplémentaires n'ont pas dépassé ce modèle. On garde donc ce
U-Net comme baseline officielle et on évite de lancer immédiatement de nouveaux
sweeps U-Net simples.

## Pourquoi changer d'architecture

Le U-Net local est maintenant un baseline solide: splits no-leak, CE-Dice,
pondérations de classes, augmentation, sampler, 100/250/500 epochs et TTA ont
été testés. La prochaine amélioration raisonnable doit venir d'une meilleure
exploitation de la structure `avant/après`, pas seulement d'un autre réglage
mineur.

## Priorité des architectures

1. `siamese_unet_shared_encoder`
   - entrée 6 canaux divisée en `pre` et `post`;
   - encodeur partagé pour les deux images;
   - fusion multi-niveaux par `concat(pre_feat, post_feat, abs(post_feat - pre_feat))`;
   - décodeur U-Net vers les 3 classes de dégâts.

2. Variante change-aware Siamese
   - la première version est déjà change-aware via les différences de features;
   - une branche explicite d'entrée `abs(post - pre)` reste une évolution possible,
     mais elle n'est pas incluse dans la vague 1 pour limiter mémoire et complexité.

TODO Siamese U-Net++: une version nested decoder peut être pertinente, mais elle
n'est pas implémentée dans cette première passe. La priorité est de valider
d'abord le modèle Siamese partagé simple avec un smoke test et une vague courte.

3. DeepLabV3+ 6 canaux
   - `smp_deeplabv3plus_resnet50_6ch`;
   - `smp_deeplabv3plus_effb3_6ch`.

4. SMP U-Net 6 canaux
   - `smp_unet_effb3_6ch`;
   - `smp_unet_resnet50_6ch`.

5. Architectures transformer/change
   - SegFormer et ChangeFormer sont prometteurs, mais reportés à une vague
     ultérieure pour éviter une surface de dépendances trop large avant d'avoir
     validé les modèles CNN/Siamese.

## Fichiers ajoutés

```text
src/crisismap/models/siamese_unet.py
src/crisismap/models/damage_model_factory.py
scripts/train_damage_architecture.py
scripts/evaluate_damage_architecture.py
slurm/smoke_damage_architecture.sbatch
slurm/run_damage_arch_config.sh
slurm/submit_damage_arch_sweep_v1.sh
configs/damage_arch_sweep_v1.csv
```

Le fichier `train_unet.py` n'est pas modifié. Le pipeline Axis 2 est séparé pour
préserver le comportement du baseline existant.

## Vague 1

La première vague est volontairement compacte:

1. Siamese shared encoder, `splits_noleak_match_hist_all`, safe, sqrt alpha 4.
2. Siamese shared encoder, `splits_noleak_match_hist1000`, safe, sqrt alpha 4.
3. Siamese shared encoder, `splits_noleak_match_hist_all`, damage-aware, sqrt alpha 4.
4. Siamese shared encoder, `splits_noleak_match_hist_all`, safe, sqrt alpha 8.
5. DeepLabV3+ ResNet50 6 canaux, `splits_noleak_match_hist_all`.
6. DeepLabV3+ EfficientNet-B3 6 canaux, `splits_noleak_match_hist_all`.
7. SMP U-Net EfficientNet-B3 6 canaux, `splits_noleak_match_hist_all`.
8. SMP U-Net ResNet50 6 canaux, `splits_noleak_match_hist_all`.

Toutes les lignes sont définies dans:

```text
configs/damage_arch_sweep_v1.csv
```

## Protocole de comparaison

Les modèles doivent être comparés avec:

- splits no-leak;
- image size `1024`;
- target mode `3-class`;
- loss `ce-dice`;
- class weights `0.05 1.0 4.0`;
- augmentation `safe` ou `damage-aware`;
- sampler `damage-sqrt`;
- métriques: mean IoU, IoU damaged, precision damaged, recall damaged, F1 damaged;
- première vague: `100 epochs`;
- finalistes: `250 epochs`;
- TTA séparée après entraînement, pas pendant l'évaluation standard.

Note TTA: `scripts/evaluate_damage_tta.py` reste l'outil validé pour le U-Net
champion. Pour les nouvelles architectures, la première étape est l'évaluation
standard avec `scripts/evaluate_damage_architecture.py`; la TTA sera étendue au
model factory uniquement pour les architectures finalistes afin d'éviter de
dupliquer trop tôt de la logique.

## Smoke test avant lancement

Avant tout entraînement:

```bash
sbatch slurm/smoke_damage_architecture.sbatch
```

Ce job instancie chaque architecture du CSV sur H100 et exécute un forward dummy
`[1, 6, 256, 256]`. Il ne lit pas les données et n'écrit pas de checkpoints.

## Lancement du sweep, plus tard

Quand le smoke test est validé:

```bash
bash slurm/submit_damage_arch_sweep_v1.sh
```

Le submitter lance une tâche indépendante par ligne du CSV. Il n'y a pas de
dépendance par défaut et aucun entraînement n'est lancé automatiquement par le
simple ajout de ces fichiers.

## Relation avec Building100

Le sweep Building100 reste séparé. Il cherche un meilleur masque binaire
bâtiment. Une fois les meilleurs modèles Axis 2 identifiés, on pourra combiner:

1. meilleur modèle damage;
2. meilleur modèle building;
3. clipping ou majorité par composante;
4. TTA `d4` si elle reste bénéfique.
