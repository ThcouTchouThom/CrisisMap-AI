# Résultats — oracle avec masque de bâtiments parfait

## Contexte

Cette expérience évalue le gain théorique obtenu si le modèle disposait d'une
segmentation parfaite des bâtiments au moment de l'évaluation.

Il s'agit d'une expérience **oracle** : elle utilise le masque vérité terrain
des bâtiments uniquement pour post-traiter les prédictions. Ce n'est donc pas un
résultat de production et cela ne correspond pas à un système déployable tel
quel.

## Configuration évaluée

- Checkpoint évalué :
  `unet_1024_ce_dice_w005_1_4_noleak_match_hist1000_bs2_250epochs`
- Split de test :
  `data/processed/splits_full/test_pairs.csv`
- Nombre d'échantillons évalués : `222`
- Taille d'image : `1024`
- Connectivité des composantes : `4`
- Politique pour composante vide : `no_damage`

## Modes comparés

### raw

Prédiction normale du modèle, sans modification.

### oracle_building_clip

Le masque de bâtiments vérité terrain est utilisé pour forcer toutes les
prédictions hors bâtiment à la classe `0` (`background`). À l'intérieur des
bâtiments vérité terrain, la prédiction brute du modèle est conservée.

### oracle_building_component_majority

Le masque de bâtiments vérité terrain est utilisé pour identifier les
composantes connexes correspondant aux bâtiments. Après clipping hors bâtiment,
chaque composante est convertie en une décision cohérente :

- majorité de pixels prédits `2` : bâtiment endommagé;
- sinon : bâtiment non endommagé;
- si aucune classe bâtiment n'est prédite dans une composante, la politique
  `no_damage` attribue la classe `1`.

Cette approximation est pertinente pour xBD/xView2, où un bâtiment annoté est
généralement associé à un niveau de dommage unique.

## Métriques

| Mode | Mean IoU | IoU damaged | Precision damaged | Recall damaged | F1 damaged |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw | 0.6651 | 0.4175 | 0.6307 | 0.5526 | 0.5891 |
| oracle_building_clip | 0.7385 | 0.4782 | 0.7804 | 0.5526 | 0.6470 |
| oracle_building_component_majority | 0.7997 | 0.5383 | 0.7820 | 0.6333 | 0.6999 |

## Gains par rapport au mode brut

| Mode oracle | Delta Mean IoU | Delta IoU damaged | Delta Precision damaged | Delta Recall damaged | Delta F1 damaged |
| --- | ---: | ---: | ---: | ---: | ---: |
| oracle_building_clip | +0.0734 | +0.0607 | +0.1497 | +0.0000 | +0.0579 |
| oracle_building_component_majority | +0.1346 | +0.1208 | +0.1513 | +0.0807 | +0.1108 |

## Interprétation

Le gain est important, surtout pour le mode
`oracle_building_component_majority` :

- l'IoU de la classe endommagée passe de `0.4175` à `0.5383`;
- le F1 de la classe endommagée passe de `0.5891` à `0.6999`;
- le Mean IoU passe de `0.6651` à `0.7997`.

Cela indique qu'une partie significative des erreurs vient de la confusion entre
localisation des bâtiments et classification des dommages. Le modèle ne se
trompe pas seulement sur le niveau de dommage : il perd aussi de la performance
parce que la segmentation bâtiment/fond n'est pas parfaitement maîtrisée.

## Conclusion

Ces résultats soutiennent fortement l'exploration d'un pipeline en deux étapes :

1. segmentation des bâtiments;
2. classification des bâtiments, ou des pixels bâtiment, en intact/endommage.

Le résultat reste un **borne supérieure oracle** : le masque bâtiment vérité
terrain n'est pas disponible en production. Il sert à mesurer le potentiel d'une
meilleure segmentation de bâtiments, pas à annoncer une performance finale du
système.
