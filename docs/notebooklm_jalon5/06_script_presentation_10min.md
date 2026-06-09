# Script de présentation 10 minutes

Ce document sert à aider NotebookLM à générer un script oral clair et fluide.

## Slide 1 - Titre

**Aftermath - Voir les dégâts pour agir plus vite**

Message oral :

> Aftermath est un projet de cartographie automatique des dommages après catastrophe à partir d'images satellite. L'idée est simple : comparer une image avant et une image après pour produire une carte visuelle des bâtiments intacts et endommagés.

Durée : 30 secondes.

## Slide 2 - Problème réel

Points :

- après catastrophe, il faut analyser vite ;
- les images satellite couvrent de grandes zones ;
- l'analyse manuelle est lente ;
- les décisions terrain doivent être priorisées.

Message oral :

> Après un ouragan, un incendie ou un séisme, les équipes de crise doivent comprendre rapidement où sont les dégâts. Les satellites donnent une vue large, mais il faut encore interpréter ces images. C'est exactement le problème que nous voulons accélérer.

Durée : 45 secondes.

## Slide 3 - Solution proposée

Points :

- entrée : paire pré/post catastrophe ;
- sortie : carte de dégâts ;
- classes : fond, bâtiment intact, bâtiment endommagé ;
- interface Streamlit.

Message oral :

> Notre solution prend une paire d'images satellite avant/après et génère un masque de segmentation. Le résultat est lisible : noir pour le fond, vert pour les bâtiments intacts, rouge pour les bâtiments endommagés.

Durée : 45 secondes.

## Slide 4 - Dataset

Points :

- xBD / xView2 ;
- paires pré/post ;
- catastrophes variées ;
- annotations de bâtiments et de dommages.

Message oral :

> Nous travaillons sur xBD, aussi appelé xView2, un dataset de référence pour l'évaluation des dommages sur bâtiments. Il contient des images avant et après catastrophe, avec des annotations.

Durée : 45 secondes.

## Slide 5 - Pipeline technique

Points :

- image pré/post ;
- modèle damage ;
- TTA d4 ;
- segmentation building ;
- component majority ;
- carte finale.

Message oral :

> Le pipeline final combine plusieurs briques. Le modèle damage prédit les dégâts. La TTA d4 stabilise l'inférence. Ensuite, un modèle building localise les bâtiments, puis un post-processing rend la décision plus cohérente au niveau bâtiment.

Durée : 1 minute.

## Slide 6 - Baseline forte

Points :

- U-Net 6 canaux ;
- TTA d4 ;
- F1 damaged 0.631300 ;
- IoU damaged 0.461240 ;
- mean IoU 0.681574.

Message oral :

> Nous avons d'abord construit une baseline U-Net solide. Elle n'est pas simplement un premier essai : elle a été optimisée avec une bonne stratégie d'entraînement et de TTA. Elle donne donc une vraie référence à battre.

Durée : 45 secondes.

## Slide 7 - Nouveau champion damage

Points :

- `dftv2_hist1000_attention_sqrt2_ft_250_seed0` ;
- `siamese_unet_attention` ;
- focal-Tversky ;
- F1 damaged 0.701317 ;
- IoU damaged 0.540022 ;
- mean IoU 0.728266.

Message oral :

> Le meilleur modèle actuel est une architecture Siamese Attention. Elle est plus adaptée au problème, car elle exploite explicitement la structure avant/après. Elle atteint un F1 damaged de 0.7013, contre 0.6313 pour la baseline U-Net avec TTA.

Durée : 1 minute.

## Slide 8 - Segmentation bâtiment

Points :

- modèle building `b400_effb4_sampler8_ft` ;
- U-Net++ EfficientNet-B4 ;
- F1 building 0.850421 ;
- IoU building 0.739767 ;
- rôle : contraindre la carte damage.

Message oral :

> Nous avons aussi entraîné un modèle dédié à la segmentation bâtiment. Il permet de mieux localiser les structures, puis d'appliquer une décision plus cohérente par bâtiment. C'est une étape importante pour passer d'une carte pixel-level à une logique plus proche du terrain.

Durée : 1 minute.

## Slide 9 - Application de démonstration

Points :

- Streamlit ;
- mode dataset ;
- mode upload ;
- trois pipelines ;
- visualisation brut, building, final ;
- overlay.

Message oral :

> L'application Streamlit permet de lancer le pipeline sur un exemple du dataset ou sur une paire téléversée. Elle affiche les images avant/après, la prédiction damage brute, le masque bâtiment, la prédiction finale et l'overlay.

Durée : 1 minute.

## Slide 10 - Démonstration live

Déroulé :

1. choisir le champion damage ;
2. choisir le pipeline qualité maximale ;
3. charger une paire ou sélectionner un exemple ;
4. lancer l'inférence ;
5. montrer l'overlay final ;
6. montrer les panneaux explicatifs.

Message oral :

> Ici, on voit que l'application ne sort pas seulement une image colorée. Elle expose aussi les étapes intermédiaires : damage brut, masque bâtiment et damage post-processé. Cela rend le résultat plus explicable.

Durée : 1 minute 30.

## Slide 11 - Limites

Points :

- modèle encore imparfait ;
- 3 classes seulement ;
- dépendance aux images satellite ;
- masque building parfois imparfait ;
- besoin de validation humaine.

Message oral :

> Le système reste un prototype. Il peut se tromper, notamment quand les dégâts sont subtils ou quand le bâtiment est mal segmenté. L'objectif est une aide à la décision, pas une décision automatique finale.

Durée : 45 secondes.

## Slide 12 - Roadmap et conclusion

Points :

- TTA et ensembles ;
- amélioration building ;
- passage 5 classes ;
- intégration SIG ;
- produit de priorisation.

Message oral :

> La prochaine étape est de consolider les meilleurs modèles, tester les ensembles, améliorer le building et revenir vers la formulation xView2 5 classes. À terme, Aftermath peut devenir un outil de priorisation visuelle pour les équipes de crise.

Durée : 45 secondes.

## Conclusion courte

> Aftermath montre qu'un pipeline IA peut transformer une paire d'images satellite avant/après en une carte de dégâts lisible. Nous avons progressé d'une baseline U-Net vers un champion Siamese Attention plus performant, intégré un modèle bâtiment, et construit une application de démonstration capable d'expliquer visuellement ses sorties.
