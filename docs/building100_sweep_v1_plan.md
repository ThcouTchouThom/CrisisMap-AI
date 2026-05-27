# Plan Building100 Sweep v1 - Segmentation bâtiment

## Objectif

La première expérience locale building-only a montré qu'un segmentateur bâtiment dédié est prometteur :

| Élément | Valeur |
| --- | --- |
| Modèle | U-Net++ EfficientNet-B3 |
| Entrée | pré-catastrophe |
| Cible | `building-binary`, soit `target > 0` |
| Loss | focal Tversky |
| Validation building IoU | `0.669204` |
| Validation F1 | `0.801824` |
| Test building IoU | `0.658813` |
| Test F1 | `0.794318` |
| Test precision | `0.809546` |
| Test recall | `0.779653` |

La campagne Building100 vise à explorer largement les modèles, pertes, entrées, augmentations, samplers, learning rates et splits avant de sélectionner quelques finalistes pour des entraînements longs à 250 ou 400 epochs.

## Pourquoi une campagne plus large

La proposition initiale de 12 runs était utile pour confirmer rapidement l'intérêt du building-only. Elle est cependant trop étroite pour choisir sérieusement un modèle final, car elle ne teste pas :

- plusieurs familles d'architectures ;
- plusieurs losses adaptées au binaire ;
- l'entrée `pre-post` ;
- la sensibilité aux augmentations ;
- la sensibilité au sampler ;
- la sensibilité au learning rate ;
- l'effet du split d'entraînement.

Building100 v1 reste structurée, mais couvre un espace plus riche avec environ **49 runs**.

## Organisation config-driven

La campagne est pilotée par :

```text
configs/building100_sweep_v1.csv
slurm/run_building100_config.sh
slurm/submit_building100_sweep_v1.sh
scripts/rebuild_building100_summary.py
```

Le CSV contient une ligne par expérience. Le script de soumission lit le CSV et soumet un job SLURM indépendant par ligne.

Par défaut, les jobs attendent la fin des jobs damage long250 via :

```text
afterany:13271923:13271924:13271925:13271926:13271927
```

Pour soumettre sans dépendance :

```bash
WAIT_FOR_LONG250=0 bash slurm/submit_building100_sweep_v1.sh
```

## Blocs expérimentaux

### Bloc A - Architecture × loss sur le grand split no-leak

Split :

```text
data/processed/splits_noleak_full_train/
```

Entrée : `pre`

Augmentation : `building-safe`

Sampler : `none`

Learning rate : `1e-4`

Modèles :

- `unet`
- `unetplusplus_effb3`
- `unetplusplus_effb4`
- `deeplabv3plus_resnet50`
- `deeplabv3plus_effb3`
- `fpn_effb3`

Losses :

- `bce-dice`
- `focal-dice`
- `focal-tversky`

But : identifier rapidement les familles modèle/loss les plus efficaces.

### Bloc B - Entrée pre-post

Entrée : `pre-post`

Loss : `focal-tversky`

Augmentation : `building-safe`

Sampler : `none`

But : vérifier si l'image post-catastrophe apporte de l'information pour segmenter les bâtiments, ou si l'image pré-catastrophe suffit.

### Bloc C - Sensibilité aux augmentations

Entrée : `pre`

Loss : `focal-tversky`

Sampler : `none`

Augmentations ajoutées :

- `none`
- `building-strong`

La variante `building-safe` est déjà couverte par le bloc A.

But : savoir si l'augmentation aide réellement les contours bâtiment ou si elle dégrade les détails fins.

### Bloc D - Sensibilité au sampler

Sampler :

```text
building-sqrt
sample_weight = 1 + alpha * sqrt(building_ratio)
```

Alphas :

- `4`
- `8`

But : améliorer le rappel bâtiment sans trop dégrader la précision.

### Bloc E - Sensibilité au learning rate

Learning rates ajoutés :

- `5e-5`
- `2e-4`

Le learning rate `1e-4` est déjà couvert par le bloc A.

But : distinguer les modèles vraiment faibles des modèles simplement mal réglés.

### Bloc F - Sensibilité au split d'entraînement

Modèle : `unetplusplus_effb3`

Entrée : `pre`

Loss : `focal-tversky`

Augmentation : `building-safe`

Splits testés :

- `splits_noleak_match_hist1000`
- `splits_noleak_match_hist_all`
- `splits_noleak_building_rich_002`
- `splits_noleak_dmg001_v2`

But : vérifier si un split conçu pour le damage reste adapté à la segmentation bâtiment.

## Métriques

L'évaluation teste les seuils :

```text
0.3, 0.4, 0.5, 0.6
```

Métriques principales :

1. `building_iou`
2. `building_f1`
3. `building_recall`
4. `building_precision`
5. `object_recall`
6. `object_precision`

Le seuil 0.5 reste le point de référence classique. Les seuils 0.3 et 0.4 sont importants pour le pipeline damage, car un masque bâtiment plus permissif peut améliorer le rappel et éviter de supprimer de vrais bâtiments avant la classification des dommages.

## Résumé attendu

Après chaque run terminé, le script reconstruit :

```text
outputs/predictions/building100_sweep_v1_summary.csv
```

Ce résumé contient :

- configuration complète ;
- chemin du checkpoint ;
- chemin des métriques ;
- meilleures métriques validation ;
- métriques test aux seuils 0.3, 0.4, 0.5, 0.6 ;
- meilleur seuil par building IoU ;
- meilleur seuil par F1 ;
- meilleur seuil par recall ;
- métriques objet si disponibles.

## Sélection des finalistes

Les finalistes pour 250 ou 400 epochs seront choisis selon :

- building IoU élevé ;
- F1 élevé ;
- rappel suffisant, surtout à seuil 0.3 ou 0.4 ;
- précision pas trop dégradée ;
- stabilité validation/test ;
- coût d'entraînement raisonnable.

Un modèle peut être retenu même s'il n'a pas le meilleur IoU à seuil 0.5, s'il offre un meilleur rappel à seuil plus bas pour le futur post-processing damage.

## Pourquoi SegFormer est différé

SegFormer est une architecture pertinente pour la segmentation satellite, mais elle ajoute une nouvelle famille d'implémentation et de dépendances. Pour cette vague, l'objectif est de rester dans `segmentation_models_pytorch` et dans les modèles déjà propres à intégrer :

- U-Net local ;
- U-Net++ ;
- DeepLabV3+ ;
- FPN.

SegFormer pourra faire l'objet d'une vague séparée si la campagne Building100 confirme que la segmentation bâtiment améliore vraiment le pipeline damage.

## Pourquoi l'augmentation faux nuages est différée

Les faux nuages pourraient simuler des artefacts réalistes, mais ils risquent aussi de masquer artificiellement des bâtiments et de rendre l'interprétation plus difficile. La campagne v1 se limite donc à des transformations sûres :

- flips ;
- rotations 90 degrés ;
- luminosité/contraste ;
- gamma ;
- bruit léger ;
- flou léger.

Les nuages synthétiques pourront être testés plus tard, une fois un baseline building solide établi.

## Relaunch après timeout / correction environnement

Diagnostic Rorqual :

- l'environnement est maintenant corrigé : `timm==1.0.27` et `segmentation_models_pytorch==0.5.0` sont installés dans le venv ;
- `torch` s'importe correctement ;
- `torch.cuda.is_available()` peut être `False` sur le login node, ce qui est normal. CUDA doit être vérifié dans un job SLURM GPU ;
- les trois runs U-Net locaux ont timeout vers 80-82 epochs et possèdent des `last_building.pt` ;
- les autres jobs SMP ont échoué avant entraînement lorsque les dépendances manquaient.

Le script suivant vérifie l'environnement sans installer automatiquement quoi que ce soit :

```bash
python scripts/check_rorqual_building_env.py
```

Relance recommandée :

```bash
bash slurm/submit_building100_resume_missing.sh
```

Cette commande utilise `configs/building100_sweep_v1_relaunch.csv` avec des limites de temps plus longues :

- `unet` : `05:00:00`
- `fpn_effb3` : `06:00:00`
- `unetplusplus_effb3` : `07:00:00`
- `deeplabv3plus_resnet50` : `07:00:00`
- `deeplabv3plus_effb3` : `08:00:00`
- `unetplusplus_effb4` : `09:00:00`
- ajout de `+01:00:00` pour `pre-post`, `building-strong` ou les variantes avec sampler.

Règles de sécurité :

- run complet + JSON/CSV test présents : skip ;
- historique complet mais métriques test absentes : évaluation seulement ;
- run incomplet + `RESUME_INCOMPLETE=1` + `last_building.pt` : reprise ;
- run manquant : entraînement normal ;
- run incomplet sans option explicite : arrêt propre, pas d'évaluation ;
- `FORCE_INCOMPLETE=1` supprime seulement le run incomplet visé avant réentraînement.

Les métriques officielles ne sont produites qu'après entraînement complet et évaluation test.

Audit :

```bash
python scripts/audit_campaign_completion.py --campaign building100
```
