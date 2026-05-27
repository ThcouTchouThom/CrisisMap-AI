# Plan campagne long250 - Augmentation et sampler

## Objectif

La campagne 2 à 100 epochs a terminé ses 32 runs Rorqual. Le meilleur résultat est très proche du champion no-leak précédent, mais ne le dépasse pas clairement :

| Modèle | Mean IoU | IoU damaged | F1 damaged |
| --- | ---: | ---: | ---: |
| Champion no-leak 250 epochs | `≈ 0.665062` | `≈ 0.417517` | `≈ 0.589082` |
| Meilleur Campaign 2, 100 epochs | `0.650122` | `0.416751` | `0.588319` |

Cette proximité suggère que certaines configurations de Campaign 2 pourraient dépasser le champion avec un entraînement plus long. La campagne long250 compare donc cinq configurations ciblées à **250 epochs**.

## Configurations retenues

### 1. Reproduction du champion

- split : `splits_noleak_match_hist1000`
- augmentation : `none`
- sampler : `none`
- rôle : contrôle expérimental

Cette configuration permet de vérifier que le protocole long250 reproduit correctement le champion précédent.

### 2. Même split que le champion, avec augmentation damage-aware

- split : `splits_noleak_match_hist1000`
- augmentation : `damage-aware`
- sampler : `none`
- rôle : isoler l'effet de `damage-aware`

Ce run teste si l'amélioration observée à 100 epochs vient vraiment de l'augmentation ou surtout du split.

### 3. Meilleur candidat global à 100 epochs

- split : `splits_noleak_match_hist_all`
- augmentation : `damage-aware`
- sampler : `none`
- rôle : meilleur compromis IoU/F1 de Campaign 2

C'est le candidat principal pour dépasser le champion précédent.

### 4. Candidat haut rappel

- split : `splits_noleak_dmg001_v2`
- augmentation : `damage-aware`
- sampler : `none`
- rôle : améliorer le rappel damaged

Ce split favorise davantage les exemples avec dommages. Il peut être utile si l'objectif est de limiter les faux négatifs, quitte à accepter plus de faux positifs.

### 5. Candidat haute sensibilité

- split : `splits_noleak_match_hist_all`
- augmentation : `safe`
- sampler : `damage-sqrt`
- alpha : `4`
- rôle : maximiser le rappel damaged

Cette configuration était le meilleur mode haut rappel à 100 epochs. Elle est moins équilibrée, mais peut devenir pertinente avec un post-processing bâtiment.

## Configuration commune

- modèle : U-Net damage
- image size : `1024`
- batch size : `2`
- loss : `ce-dice`
- class weights : `0.05 1.0 4.0`
- learning rate : `1e-4`
- epochs : `250`
- validation : common validation no-leak via les splits utilisés
- test : `data/processed/splits_full/test_pairs.csv`

## Organisation Rorqual

Un fichier `.sbatch` est créé pour chaque configuration. Les jobs sont indépendants et ne déclarent pas de dépendance par défaut.

Helper optionnel :

```bash
bash slurm/submit_long250_aug_sampler_campaign.sh
```

Ce helper soumet les cinq jobs et affiche les IDs retournés par `sbatch`.

Résumé final attendu :

```text
outputs/predictions/unet_1024_long250_aug_sampler_summary.csv
```

Le résumé est reconstruit après chaque job terminé afin de ne pas perdre les résultats si une autre tâche échoue ou dépasse son temps limite.

## Métriques prioritaires

Les modèles doivent être comparés en priorité sur :

1. `IoU damaged`
2. `F1 damaged`
3. `Recall damaged`
4. `Precision damaged`
5. `Mean IoU`

L'IoU damaged reste la métrique principale, car elle mesure la qualité de segmentation de la classe la plus rare et la plus importante pour l'usage métier. Le rappel est aussi important, mais il doit être interprété avec la précision pour éviter de sélectionner un modèle qui détecte trop de faux dommages.

## Critère de décision

Un nouveau modèle devient candidat champion s'il dépasse le précédent sur :

- `IoU damaged`, ou
- `F1 damaged` avec rappel nettement supérieur sans chute excessive de précision.

Si aucun run ne dépasse clairement le champion 250 epochs, la conclusion la plus probable sera que l'amélioration doit venir d'une autre source : post-processing bâtiment, segmentation bâtiment dédiée, architecture plus forte ou retour vers une formulation multi-tâches.
