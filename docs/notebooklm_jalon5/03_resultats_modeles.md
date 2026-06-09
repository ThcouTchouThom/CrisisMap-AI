# Résultats des modèles

## Métriques principales

Pour la classe damage, les métriques les plus importantes sont :

- **IoU damaged** : mesure la qualité de recouvrement des bâtiments endommagés ;
- **F1 damaged** : équilibre entre précision et rappel ;
- **mean IoU** : performance moyenne sur les classes.

La classe damage est la plus critique, car elle correspond aux zones à prioriser.

## Baseline forte U-Net

La baseline initiale a été poussée assez loin pour devenir une vraie référence.

Pipeline :

- U-Net ;
- entrée 6 canaux ;
- image size 1024 ;
- no-leak protocol ;
- augmentation ;
- sampler damage ;
- TTA d4.

Résultats :

| Modèle | F1 damaged | IoU damaged | Mean IoU |
| --- | ---: | ---: | ---: |
| U-Net + TTA d4 | 0.631300 | 0.461240 | 0.681574 |

Cette baseline est importante car elle donne un point de comparaison solide. Les nouveaux modèles ne doivent pas seulement battre un modèle faible, mais une baseline déjà optimisée.

## Ancien champion

Avant la campagne focal-Tversky v2, le meilleur modèle était :

`dlong100_hist1000_attention_safe_sqrt4_focal_tversky`

Résultats :

| Modèle | F1 damaged | IoU damaged | Mean IoU |
| --- | ---: | ---: | ---: |
| Ancien champion Siamese Attention | 0.678801 | 0.513776 | 0.707285 |

Ce résultat a montré que la famille Siamese Attention était très prometteuse pour le problème xBD, car elle exploite explicitement la structure pré/post.

## Nouveau champion actuel

Le nouveau meilleur modèle est :

`dftv2_hist1000_attention_sqrt2_ft_250_seed0`

Configuration :

| Élément | Valeur |
| --- | --- |
| Architecture | `siamese_unet_attention` |
| Loss | `focal-tversky` |
| Split | `hist1000` |
| Sampler | `damage-sqrt alpha2` |
| Epochs | 250 |

Résultats :

| Modèle | F1 damaged | IoU damaged | Mean IoU |
| --- | ---: | ---: | ---: |
| Nouveau champion damage | 0.701317 | 0.540022 | 0.728266 |

## Comparaison synthétique

| Modèle | F1 damaged | IoU damaged | Mean IoU |
| --- | ---: | ---: | ---: |
| U-Net + TTA d4 | 0.631300 | 0.461240 | 0.681574 |
| Ancien champion Siamese | 0.678801 | 0.513776 | 0.707285 |
| Nouveau champion Siamese | 0.701317 | 0.540022 | 0.728266 |

## Lecture des gains

Par rapport à la baseline U-Net + TTA d4 :

- F1 damaged passe de 0.631300 à 0.701317 ;
- IoU damaged passe de 0.461240 à 0.540022 ;
- mean IoU passe de 0.681574 à 0.728266.

Le gain est significatif parce qu'il porte sur la classe endommagée, qui est la classe la plus difficile et la plus importante.

## Champion building

Le meilleur modèle building actuel est :

`b400_effb4_sampler8_ft`

Configuration :

| Élément | Valeur |
| --- | --- |
| Architecture | U-Net++ EfficientNet-B4 |
| Tâche | Segmentation bâtiment binaire |

Résultats :

| Modèle | F1 building | IoU building |
| --- | ---: | ---: |
| b400 EffB4 | 0.850421 | 0.739767 |

## Pourquoi le modèle building compte

Le modèle building sert à contraindre le résultat damage. Il aide à :

- réduire les faux positifs hors bâtiments ;
- rendre la prédiction plus cohérente au niveau objet ;
- rapprocher le pipeline d'une logique métier : localiser les bâtiments puis qualifier leur état.

## Interprétation globale

La trajectoire expérimentale est claire :

1. U-Net simple : base fonctionnelle ;
2. U-Net optimisé : baseline forte ;
3. Siamese Attention : exploitation du pré/post ;
4. focal-Tversky et sampler : meilleure gestion de la rareté du damage ;
5. building segmentation : meilleure cohérence spatiale.

## Message pour l'oral

Le point clé est que nous ne nous sommes pas arrêtés à une première baseline. Nous avons construit une baseline forte, puis nous avons montré qu'une architecture plus adaptée au problème pré/post, la Siamese Attention, améliore fortement la détection des bâtiments endommagés.
