# Plan d'evaluation des ensembles damage

## Objectif

Ce module ajoute une evaluation generale des ensembles de modeles pour la prediction des degats dans CrisisMap AI / Aftermath. Il ne lance aucun entrainement et ne modifie aucun checkpoint existant. L'objectif est de comparer proprement plusieurs strategies :

- combiner le champion U-Net avec des modeles Siamese / change-aware;
- tester des familles plus recentes, si leurs checkpoints existent;
- ajouter une contrainte optionnelle par masque de batiment predit;
- mesurer l'effet d'un biais sur le logit de la classe `damaged`;
- evaluer l'apport de la TTA (`none`, `flips`, `rot90`, `d4`).

## Fichier principal

Script :

```bash
python scripts/evaluate_damage_model_ensemble.py
```

Configuration des candidats :

```text
configs/damage_ensemble_candidates.csv
```

Le CSV est volontairement simple et plat. Les colonnes principales sont :

- `enabled` : `1` pour activer, `0` pour ignorer;
- `name` : nom lisible du candidat;
- `role` : `damage`, `building` ou `both`;
- `family` : famille de chargement du modele;
- `model` : nom exact du modele dans la factory correspondante;
- `checkpoint` : chemin du checkpoint;
- `weight` : poids utilise pour les moyennes ponderees;
- `label_mode` : `3-class`, `building-damage-2class`, `multilabel` ou futur `5-class`;
- `input_mode` : surtout utile pour les segmentateurs batiment (`pre`, `post`, `pre-post`).

Les checkpoints manquants sont ignores avec un avertissement clair. Cela permet de garder dans le CSV des candidats prevus mais pas encore produits.

## Familles supportees

Le script supporte actuellement :

- `local_unet_existing` : champion U-Net existant;
- `siamese_unet_attention`;
- `siamese_unet_abs_signed`;
- `multitemporal_fusion`;
- `xview2_strong_baseline`;
- `multihead_damage`;
- `building_segmentation` pour fournir uniquement des masques batiment.

Tous les modeles damage sont ramenes a des logits comparables en formulation 3 classes :

- `0 background`;
- `1 building no-damage`;
- `2 damaged building`.

Les sorties multi-head avec `building_logits` et `damage_logits` 1 ou 2 canaux sont converties en probabilites 3 classes avant comparaison. Les futurs modeles 5 classes peuvent etre collapses en `no_damage` et `damaged`, mais ces scores ne doivent pas etre presentes comme scores officiels xView2.

## Modes d'ensemble

Modes disponibles :

- `average_logits` : moyenne simple des logits;
- `average_prob` : moyenne des probabilites softmax;
- `weighted_average_logits` : moyenne ponderee des logits;
- `weighted_average_prob` : moyenne ponderee des probabilites;
- `majority_vote` : vote majoritaire pixel par pixel.

Le parametre `--damage-biases` permet d'ajouter un biais au logit de la classe `damaged`. Il faut l'utiliser prudemment : un biais choisi sur le test n'est pas une selection valide. Pour une comparaison rigoureuse, le biais doit etre choisi sur validation puis applique au test.

## TTA

Le script applique la TTA a chaque candidat avant l'ensemble :

- `none`;
- `flips`;
- `rot90`;
- `d4`.

Les logits sont inverses dans le repere original avant d'etre moyennes. L'argmax est applique seulement apres l'ensemble.

## Contraintes par masque batiment

Modes disponibles :

- `none`;
- `predicted_building_clip`;
- `predicted_building_component_majority`;
- `building_ensemble_mask`.

Ces modes utilisent seulement des candidats qui fournissent `building_logits` ou un modele `building_segmentation`. Aucun masque oracle n'est utilise. Si aucun candidat batiment n'est charge, ces modes sont ignores.

Interpretation :

- `clip` peut augmenter la precision en supprimant des faux positifs hors batiments;
- `component_majority` force une decision coherente par composante de batiment predit;
- ces contraintes peuvent aussi baisser le rappel si le segmentateur batiment manque des zones endommagees.

## Exemple de commande

```powershell
python scripts\evaluate_damage_model_ensemble.py `
  --candidates-csv configs\damage_ensemble_candidates.csv `
  --root data\raw\xbd\train `
  --split-csv data\processed\splits_noleak_match_hist_all\test_pairs.csv `
  --image-size 1024 `
  --batch-size 2 `
  --target-mode 3-class `
  --device cuda `
  --amp `
  --num-workers 0 `
  --tta-modes none d4 `
  --ensemble-modes average_logits average_prob weighted_average_logits majority_vote `
  --damage-biases -0.2 -0.1 0.0 0.1 `
  --building-constraints none predicted_building_clip predicted_building_component_majority `
  --building-thresholds 0.5 0.6 0.7 `
  --output-json outputs\predictions\damage_model_ensemble.json `
  --output-csv outputs\predictions\damage_model_ensemble.csv `
  --ranked-csv outputs\predictions\damage_model_ensemble_ranked.csv
```

## Metriques

Le CSV de sortie contient notamment :

- `test_mean_iou`;
- `test_iou_damaged`;
- `test_precision_damaged`;
- `test_recall_damaged`;
- `test_f1_damaged`.

Le classement est fait par :

1. F1 damaged;
2. IoU damaged;
3. mean IoU.

## Precautions

- Ne pas comparer un biais optimise sur test comme un vrai resultat final.
- Ne pas melanger des checkpoints issus de splits incompatibles sans le signaler.
- Garder le protocole no-leak : validation et test communs, train sans overlap.
- Les contraintes batiment predites ne sont pas oracle et peuvent degrader le rappel.
- Les checkpoints lourds restent hors Git; le CSV peut contenir des chemins absents qui seront simplement ignores.
