# Pipeline technique

## Vue d'ensemble

Le pipeline actuel de Aftermath fonctionne en plusieurs blocs :

1. image pré-catastrophe ;
2. image post-catastrophe ;
3. modèle damage ;
4. TTA d4 ;
5. modèle building ;
6. post-processing par composante bâtiment ;
7. carte finale ;
8. visualisation Streamlit.

L'objectif est de combiner deux informations :

- le changement visible entre avant et après ;
- la localisation probable des bâtiments.

## Entrée du modèle damage

Le modèle damage reçoit une image à 6 canaux :

- RGB avant catastrophe ;
- RGB après catastrophe.

Cela permet au réseau de comparer directement l'état avant et l'état après.

## Sortie damage

La sortie actuelle est une segmentation en 3 classes :

| Classe | Signification |
| --- | --- |
| 0 | Fond |
| 1 | Bâtiment intact |
| 2 | Bâtiment endommagé |

Le masque final peut être colorisé :

- fond en noir ;
- bâtiment intact en vert ;
- bâtiment endommagé en rouge.

## TTA d4

La TTA, ou test-time augmentation, consiste à faire plusieurs inférences sur des variantes géométriques de la même image, puis à moyenner les logits.

Le mode **d4** utilise :

- rotations ;
- flips ;
- inversion des transformations ;
- moyenne des logits avant argmax.

Cela stabilise la prédiction sans réentraîner le modèle.

## Modèle building

En parallèle, un modèle building prédit un masque binaire :

- fond ;
- bâtiment.

Le champion building actuel est :

| Élément | Valeur |
| --- | --- |
| Run | `b400_effb4_sampler8_ft` |
| Architecture | U-Net++ EfficientNet-B4 |
| F1 building | 0.850421 |
| IoU building | 0.739767 |

Ce modèle utilise l'image pré-catastrophe, car la géométrie des bâtiments est souvent plus claire avant l'événement.

## Post-processing par masque bâtiment

Le post-processing utilise le masque building prédit pour rendre la carte damage plus cohérente.

Principe :

1. on prédit d'abord le damage brut ;
2. on prédit le masque bâtiment ;
3. on identifie les composantes connexes du masque bâtiment ;
4. pour chaque bâtiment prédit, on regarde la majorité des pixels damage ;
5. le bâtiment entier est rempli comme intact ou endommagé ;
6. l'extérieur du masque bâtiment revient au fond.

Ce mécanisme est appelé **component majority**.

## Pourquoi cette stratégie

Le modèle damage peut parfois prédire du rouge hors bâtiment, ou fragmenter un même bâtiment entre intact et endommagé. Le masque building aide à rendre la décision plus structurée.

La logique produit est simple :

- d'abord localiser les bâtiments ;
- ensuite décider si chaque bâtiment est intact ou endommagé.

## Architecture damage gagnante

Le nouveau champion damage utilise une architecture **Siamese Attention**.

Idée :

- traiter l'image avant et l'image après avec une structure adaptée au pré/post ;
- mettre l'accent sur les différences entre les deux moments ;
- utiliser une fusion avec attention pour mieux exploiter les zones importantes.

Run actuel :

`dftv2_hist1000_attention_sqrt2_ft_250_seed0`

Caractéristiques :

- architecture : `siamese_unet_attention` ;
- loss : `focal-tversky` ;
- split : `hist1000` ;
- sampler : `damage-sqrt alpha2` ;
- epochs : 250.

## Pourquoi focal-Tversky

Le dommage est une classe rare. Beaucoup de pixels sont du fond ou des bâtiments intacts. Une loss classique peut donc trop favoriser les classes majoritaires.

La focal-Tversky aide à mieux gérer :

- le déséquilibre de classes ;
- les faux négatifs sur les bâtiments endommagés ;
- la sensibilité à la classe damage.

## Résumé oral

Notre pipeline combine un modèle damage spécialisé pré/post, une stabilisation par TTA d4, puis un modèle building qui rend la carte finale plus cohérente au niveau bâtiment. C'est ce qui transforme une segmentation brute en une carte plus exploitable pour une démonstration terrain.
