# Limites et roadmap

## Limites actuelles

Aftermath progresse fortement, mais le système reste un prototype de recherche.

Les limites principales sont :

- la formulation actuelle est en 3 classes, pas encore en 5 classes xView2 officielles ;
- certaines prédictions restent sensibles aux textures ambiguës ;
- la classe damage est rare et difficile ;
- les contours de bâtiments peuvent être imprécis ;
- le masque building peut retirer de vrais pixels endommagés s'il manque un bâtiment ;
- l'interface ne remplace pas une validation terrain ;
- les performances dépendent de la qualité et de la résolution des images satellite.

## Limites liées au dataset

xBD / xView2 est très utile, mais il a ses contraintes :

- les catastrophes sont variées ;
- les styles d'imagerie diffèrent ;
- les annotations peuvent être difficiles à aligner parfaitement ;
- certains dégâts sont visuellement subtils ;
- les classes d'origine sont plus fines que notre formulation actuelle.

La version actuelle regroupe les niveaux de dégâts en :

- bâtiment intact ;
- bâtiment endommagé.

Le passage aux 5 classes xView2 reste une étape future.

## Limites de métriques

Les métriques pixel-level peuvent parfois sous-estimer la qualité visuelle, car un petit décalage de contour pénalise fortement l'IoU.

Il faut donc combiner :

- métriques quantitatives ;
- inspection visuelle ;
- analyse d'erreur ;
- métriques orientées objets à terme.

## Limites produit

Dans un contexte réel, Aftermath dépendrait aussi :

- de la disponibilité rapide d'images satellite post-catastrophe ;
- de la couverture nuageuse ;
- de la résolution spatiale ;
- de la calibration entre images avant et après ;
- de l'intégration SIG ou cartographique ;
- d'une validation humaine.

## Roadmap technique

### Court terme

- stabiliser l'application de démonstration ;
- tester le nouveau champion avec TTA ;
- comparer le pipeline avec et sans building post-processing ;
- analyser les erreurs du champion Siamese Attention ;
- sélectionner les meilleurs exemples de démonstration.

### Moyen terme

- tester les ensembles de modèles ;
- combiner plusieurs familles : U-Net, Siamese, Multi-Temporal Fusion, xView2 Strong Baseline ;
- améliorer la segmentation bâtiment ;
- intégrer des métriques orientées objet ;
- renforcer l'explicabilité visuelle.

### Long terme

- passer à la formulation xView2 5 classes ;
- produire un score plus proche du protocole officiel xView2 ;
- intégrer une carte géoréférencée ;
- permettre une sortie compatible SIG ;
- construire une interface plus proche d'un produit utilisable par des équipes terrain.

## Roadmap produit

L'objectif produit peut évoluer vers :

- une plateforme d'analyse post-catastrophe ;
- une carte interactive de dégâts ;
- un outil de priorisation ;
- un assistant pour cellules de crise ;
- un module d'aide à l'assurance ou à l'inspection.

## Risques

Les principaux risques sont :

- interprétation excessive de prédictions imparfaites ;
- confiance excessive dans les sorties IA ;
- transfert difficile à de nouveaux pays ou capteurs ;
- latence entre catastrophe et disponibilité des images.

## Message oral

La limite principale est que nous ne prétendons pas fournir une vérité terrain automatique. Aftermath fournit une première estimation visuelle rapide, qui doit être utilisée comme aide à la décision et non comme décision finale.
