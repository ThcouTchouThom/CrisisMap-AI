# Résultats campagne 2 - Augmentation et sampler

## Contexte

La campagne 2 vise à tester l'effet des augmentations train-only et des stratégies de sampling pondéré sur le modèle damage U-Net 1024 no-leak.

Configuration commune :

- image size : `1024`
- batch size : `2`
- loss : `ce-dice`
- class weights : `0.05 1.0 4.0`
- learning rate : `1e-4`
- durée : `100 epochs`
- évaluation : common test no-leak

Plan expérimental :

- 4 splits :
  - `splits_noleak_match_hist1000`
  - `splits_noleak_building_rich_002`
  - `splits_noleak_match_hist_all`
  - `splits_noleak_dmg001_v2`
- 8 variantes par split :
  - `augment=none`, `sampler=none`
  - `augment=safe`, `sampler=none`
  - `augment=damage-aware`, `sampler=none`
  - `augment=none`, `sampler=damage-simple`
  - `augment=safe`, `sampler=damage-simple`
  - `augment=damage-aware`, `sampler=damage-simple`
  - `augment=safe`, `sampler=damage-sqrt`, `alpha=4`
  - `augment=damage-aware`, `sampler=damage-sqrt`, `alpha=4`

Total : **4 splits × 8 variantes = 32 runs Rorqual**.

Sources utilisées :

- `campaign2_aug_sampler_export.tgz`
- fichiers exportés de la campagne 2, notamment :
  - `campaign2_aug_sampler_completeness.csv`
  - `campaign2_aug_sampler_ranked.csv`
  - `campaign2_aug_sampler_missing_expected.csv`
  - `campaign2_aug_sampler_best_by_split.csv`

## Complétude

Les résultats exportés contiennent **32 runs** et les **32 sont détectés comme complets à 100 epochs**.

Un contrôle automatique indiquait 8 combinaisons apparemment manquantes. Ce n'est pas une absence de résultats : c'est un décalage de nommage.

Le fichier attendu cherchait le sampler sous la forme :

```text
damage-sqrt-alpha4
```

Alors que les résultats exportés utilisent :

```text
sampler = damage-sqrt
damage_sampling_alpha = 4
```

Les expériences correspondantes existent bien, avec des noms d'expérience contenant `sampler-damage-sqrt-alpha4`. La colonne `sampler` garde simplement le nom générique `damage-sqrt`.

## Meilleurs résultats

### Meilleur résultat global

| Champ | Valeur |
| --- | --- |
| Split | `splits_noleak_match_hist_all` |
| Augmentation | `damage-aware` |
| Sampler | `none` |
| Mean IoU | `0.650122` |
| IoU damaged | `0.416751` |
| Precision damaged | `0.537770` |
| Recall damaged | `0.649357` |
| F1 damaged | `0.588319` |

Ce run est le meilleur compromis global de la campagne 2 sur la classe endommagée.

### Deuxième meilleur résultat

| Champ | Valeur |
| --- | --- |
| Split | `splits_noleak_dmg001_v2` |
| Augmentation | `damage-aware` |
| Sampler | `none` |
| Mean IoU | `0.649520` |
| IoU damaged | `0.413870` |
| Precision damaged | `0.485943` |
| Recall damaged | `0.736180` |
| F1 damaged | `0.585443` |

Ce run est très intéressant pour le rappel : il détecte davantage de pixels endommagés, au prix d'une précision plus faible.

### Meilleur rappel damaged

| Champ | Valeur |
| --- | --- |
| Split | `splits_noleak_match_hist_all` |
| Augmentation | `safe` |
| Sampler | `damage-sqrt` |
| Alpha | `4` |
| IoU damaged | `0.403100` |
| Recall damaged | `0.743052` |
| F1 damaged | `0.574585` |

Cette configuration est intéressante si l'objectif prioritaire est de rater le moins possible de bâtiments endommagés. Elle baisse toutefois la précision et donc ne constitue pas le meilleur compromis global.

## Comparaison avec le champion no-leak 250 epochs

Référence précédente :

| Modèle | Mean IoU | IoU damaged | F1 damaged |
| --- | ---: | ---: | ---: |
| Champion no-leak 250 epochs | `≈ 0.665062` | `≈ 0.417517` | `≈ 0.589082` |
| Meilleur Campaign 2, 100 epochs | `0.650122` | `0.416751` | `0.588319` |

La campagne 2 **rejoint presque** le champion 250 epochs sur les métriques damaged, malgré seulement 100 epochs. Elle ne le dépasse cependant pas clairement.

Interprétation :

- l'augmentation `damage-aware` semble utile ;
- la durée d'entraînement reste probablement déterminante ;
- les meilleures configurations de campagne 2 méritent des runs longs.

## Conclusions principales

### Damage-aware augmentation

L'augmentation `damage-aware` est la conclusion la plus solide de cette campagne. Les deux meilleurs résultats utilisent :

```text
augment = damage-aware
sampler = none
```

Elle améliore la robustesse sans forcer artificiellement la distribution de sampling.

### Samplers

Les samplers augmentent souvent le rappel damaged, mais réduisent la précision.

En pratique :

- `damage-simple` n'est pas convaincant sur cette campagne ;
- `damage-sqrt` est intéressant pour un mode haut rappel ;
- aucun sampler ne bat clairement `damage-aware + no sampler` en compromis IoU/F1.

### Splits

Le meilleur split global est :

```text
splits_noleak_match_hist_all
```

Il donne le meilleur équilibre entre diversité et distribution de dommages.

Le split suivant reste intéressant :

```text
splits_noleak_dmg001_v2
```

Il favorise le rappel et peut être utile dans une stratégie où les faux positifs seraient filtrés plus tard.

Le split `splits_noleak_building_rich_002` n'est pas le meilleur pour l'IoU damaged. Il contient davantage de bâtiments, mais cela ne se traduit pas ici par une meilleure segmentation des dommages.

## Recommandations pour les prochains runs

À lancer en priorité :

1. `splits_noleak_match_hist_all`
   - `augment=damage-aware`
   - `sampler=none`
   - `250 à 400 epochs`

2. `splits_noleak_dmg001_v2`
   - `augment=damage-aware`
   - `sampler=none`
   - `250 à 400 epochs`

Option haut rappel :

3. `splits_noleak_match_hist_all`
   - `augment=safe`
   - `sampler=damage-sqrt`
   - `damage_sampling_alpha=4`
   - objectif : maximiser le rappel damaged, à comparer avec un post-processing bâtiment.

## Résumé prêt pour slide

- Campagne 2 : **32 runs Rorqual**, tous complets à **100 epochs**.
- Les 8 combinaisons “manquantes” sont un **problème de nommage**, pas des résultats absents.
- Meilleur run : `match_hist_all + damage-aware + no sampler`.
- Score meilleur run : `IoU damaged = 0.416751`, `F1 damaged = 0.588319`.
- Le résultat rejoint presque le champion no-leak 250 epochs : `IoU damaged ≈ 0.417517`, `F1 ≈ 0.589082`.
- Conclusion : `damage-aware` est utile ; les samplers augmentent le rappel mais coûtent en précision.
- Prochaine étape recommandée : runs longs `250-400 epochs` sur `match_hist_all` et `dmg001_v2` avec `damage-aware`, sans sampler.
