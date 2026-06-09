# Résumé exécutif - Jalon 5

## Projet

**Aftermath / CrisisMap AI**

**Slogan :** Voir les dégâts pour agir plus vite.

Aftermath est un projet d'intelligence artificielle appliqué à l'analyse post-catastrophe. L'objectif est de transformer des paires d'images satellite avant/après catastrophe en cartes visuelles de dégâts, afin d'aider à prioriser les interventions et à comprendre rapidement l'ampleur des dommages.

## Problème traité

Après une catastrophe naturelle, les équipes terrain, les collectivités, les ONG, les assureurs et les cellules de crise doivent comprendre rapidement quelles zones sont touchées. Or l'analyse manuelle d'images satellite est lente, coûteuse et difficile à mettre à l'échelle.

Aftermath vise à automatiser une première lecture visuelle :

- où sont les bâtiments ;
- quels bâtiments semblent intacts ;
- quels bâtiments semblent endommagés ;
- quelles zones doivent être inspectées ou priorisées.

## Données

Le projet utilise le dataset **xBD / xView2**, composé de paires d'images satellite :

- image pré-catastrophe ;
- image post-catastrophe ;
- annotations de bâtiments et de dégâts.

La formulation actuelle est une segmentation sémantique en 3 classes :

| Classe | Signification |
| --- | --- |
| 0 | Fond |
| 1 | Bâtiment intact |
| 2 | Bâtiment endommagé |

## Pipeline actuel

Le pipeline de démonstration actuel combine :

1. une paire satellite pré/post catastrophe ;
2. un modèle damage principal ;
3. une inférence avec TTA d4 pour stabiliser la prédiction ;
4. une segmentation bâtiment binaire ;
5. un post-processing par composante bâtiment ;
6. une visualisation finale dans Streamlit.

L'idée importante est que le modèle damage prédit les dégâts, tandis que le modèle building aide à contraindre la prédiction aux zones réellement bâties.

## Résultats clés

### Baseline forte U-Net + TTA d4

| Métrique | Valeur |
| --- | ---: |
| F1 damaged | 0.631300 |
| IoU damaged | 0.461240 |
| Mean IoU | 0.681574 |

### Ancien champion Siamese Attention

Run : `dlong100_hist1000_attention_safe_sqrt4_focal_tversky`

| Métrique | Valeur |
| --- | ---: |
| F1 damaged | 0.678801 |
| IoU damaged | 0.513776 |
| Mean IoU | 0.707285 |

### Nouveau champion actuel

Run : `dftv2_hist1000_attention_sqrt2_ft_250_seed0`

| Élément | Valeur |
| --- | --- |
| Architecture | `siamese_unet_attention` |
| Loss | `focal-tversky` |
| Split | `hist1000` |
| Sampler | `damage-sqrt alpha2` |
| Epochs | 250 |

| Métrique | Valeur |
| --- | ---: |
| F1 damaged | 0.701317 |
| IoU damaged | 0.540022 |
| Mean IoU | 0.728266 |

### Champion building

Run : `b400_effb4_sampler8_ft`

| Élément | Valeur |
| --- | --- |
| Architecture | U-Net++ EfficientNet-B4 |
| Tâche | Segmentation bâtiment binaire |

| Métrique | Valeur |
| --- | ---: |
| F1 building | 0.850421 |
| IoU building | 0.739767 |

## Message principal pour la présentation

Le projet a dépassé le stade du simple prototype. Aftermath dispose maintenant :

- d'un pipeline complet pré/post satellite ;
- d'une baseline U-Net solide ;
- d'un modèle damage Siamese Attention nettement meilleur ;
- d'un modèle building performant ;
- d'un prototype Streamlit capable de charger une paire réelle et de visualiser les prédictions ;
- d'une logique d'explicabilité visuelle par comparaison entre damage brut, masque bâtiment et prédiction finale.

## Phrase clé

Aftermath ne remplace pas l'expertise terrain, mais fournit une première carte de dégâts rapide, lisible et exploitable pour accélérer la prise de décision après catastrophe.
