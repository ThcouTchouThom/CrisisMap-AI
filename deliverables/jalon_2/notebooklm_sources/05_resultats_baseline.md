# Résultats baseline

## Baseline local 512

Le premier baseline complet a été entraîné localement en 512 x 512 sur le split initial old4.

| Modèle | Pixel accuracy | Mean IoU | IoU background | IoU no damage | IoU damaged | F1 damaged |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| U-Net 512 old4, 30 epochs | 0.9175 | 0.6257 | 0.9297 | 0.5602 | 0.3870 | 0.5581 |

Ce résultat valide le pipeline complet, mais la classe `damaged` reste difficile.

## Résultats no-leak plus avancés

Les expériences no-leak plus récentes ne sont pas nécessaires pour satisfaire le jalon 2, mais elles démontrent la progression méthodologique.

| Protocole | Mean IoU | IoU damaged | F1 damaged |
| --- | ---: | ---: | ---: |
| 1024 no-leak `match_hist1000`, 100 epochs | ~0.6578 | ~0.4159 | ~0.5874 |
| 1024 no-leak `match_hist1000`, 250 epochs | ~0.6651 | ~0.4175 | ~0.5891 |

Ces valeurs sont à présenter comme un état de recherche, pas comme résultat final.

## Lecture des métriques

Pixel accuracy :

- facile à interpréter ;
- peu suffisante, car le background domine.

Mean IoU :

- donne une vision plus équilibrée sur les classes.

IoU damaged :

- métrique prioritaire pour l'objectif post-catastrophe ;
- mesure la qualité de localisation des pixels endommagés.

F1 damaged :

- complète l'IoU ;
- utile pour comprendre le compromis entre faux positifs et faux négatifs.

