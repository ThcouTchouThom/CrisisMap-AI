# Vérité du prototype actuel

Ce fichier sert à éviter les hallucinations dans le deck final. Il précise ce que le prototype Aftermath fait réellement aujourd'hui, et ce qui doit être présenté uniquement comme une perspective future.

## Ce que le prototype actuel fait

Le prototype actuel permet de démontrer une chaîne complète d'aide à la décision sur une paire satellite avant/après catastrophe.

Fonctionnalités disponibles :

- upload manuel de deux images satellite :
  - image avant catastrophe;
  - image après catastrophe;
- sélection d'exemples issus du dataset quand les données locales sont disponibles;
- inférence damage avec le modèle **Siamese Attention**;
- utilisation de la **TTA d4** pour stabiliser la prédiction damage;
- segmentation bâtiment avec le champion **U-Net++ EfficientNet-B4 b400**;
- post-processing par masque bâtiment;
- post-processing par **component majority** pour rendre la décision plus cohérente par composante de bâtiment;
- visualisation de l'overlay final sur l'image post-catastrophe;
- visualisation des masques intermédiaires :
  - damage brut;
  - masque bâtiment prédit;
  - damage final post-processé;
- visualisation de l'incertitude du modèle;
- affichage de métriques quand une vérité terrain est disponible;
- exports PNG;
- export JSON.

Le prototype est donc une application de démonstration fonctionnelle : il accepte une paire d'images, lance une inférence, affiche une carte de dommages, montre les étapes intermédiaires et permet d'exporter un résultat.

## Ce que le prototype actuel ne fait pas encore

Le prototype actuel ne doit pas être présenté comme un outil SIG complet.

Il ne fait pas encore :

- export SIG complet;
- export GeoJSON;
- export GeoTIFF;
- intégration QGIS;
- intégration ArcGIS;
- API publique;
- géoréférencement opérationnel;
- workflow complet de cartographie géospatiale en production.

Ces éléments sont importants pour la vision produit, mais ils ne sont pas encore implémentés dans le prototype présenté.

## Comment présenter ces éléments

Les éléments SIG doivent être formulés comme des perspectives futures :

- "À terme, Aftermath pourrait exporter des résultats GeoJSON ou GeoTIFF."
- "Une intégration QGIS / ArcGIS fait partie de la roadmap."
- "Le géoréférencement opérationnel est une prochaine étape produit."
- "L'API publique n'est pas encore disponible, mais elle serait logique pour une version déployée."

Formulations à éviter :

- "Le prototype exporte déjà du GeoJSON."
- "L'application est intégrée à QGIS."
- "Aftermath fournit déjà un workflow SIG complet."
- "Le géoréférencement est opérationnel."

## Message à faire passer

Le bon cadrage est :

> Le prototype actuel démontre la chaîne IA et l'expérience utilisateur principale. Les exports SIG, l'API et l'intégration géospatiale complète sont des étapes futures pour passer d'un prototype bêta à un produit opérationnel.

## Résumé court pour le deck

À dire simplement :

> Aujourd'hui, Aftermath sait charger une paire satellite, lancer le pipeline damage + TTA + building, visualiser les résultats et exporter des PNG/JSON. Les exports SIG et l'intégration QGIS/ArcGIS sont dans la roadmap, pas dans la version actuelle.
