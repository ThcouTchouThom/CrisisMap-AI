# Fiche produit - Aftermath

**Slogan :** Voir les dégâts pour agir plus vite.

## Problème

Après une catastrophe naturelle, les premières heures sont critiques. Les équipes de crise doivent comprendre rapidement quelles zones sont touchées, mais l'analyse manuelle d'images satellite est lente, coûteuse et demande une expertise SIG.

## Utilisateurs cibles

- ONG;
- sécurité civile;
- cellules de crise territoriales;
- assureurs et gestionnaires de risques.

## Solution proposée

Aftermath analyse une paire d'images satellite avant/après catastrophe et produit une carte visuelle des bâtiments intacts et endommagés. L'objectif est d'accélérer la première évaluation visuelle et d'aider à prioriser les zones à inspecter.

Le prototype final est l'application classique `app/streamlit_app.py`.

## Fonctionnalités du prototype

- upload ou sélection dataset d'une paire satellite;
- exemples embarqués pour tester sans dataset complet;
- inférence damage;
- TTA d4;
- segmentation bâtiment;
- post-processing par composantes;
- overlay sur image post-catastrophe;
- visualisation des étapes intermédiaires;
- visualisation de l'incertitude;
- exports PNG / JSON.

## IA utilisée

- dataset xBD / xView2;
- segmentation sémantique pré/post catastrophe;
- architecture damage Siamese Attention;
- U-Net++ EfficientNet-B4 pour la segmentation bâtiment;
- loss focal-Tversky;
- TTA d4;
- post-processing par component majority.

## Résultats principaux

- baseline U-Net + TTA d4 : F1 damaged = **0.6313**;
- champion intégré : F1 damaged = **0.7013**, IoU damaged = **0.5400**, mean IoU = **0.7283**;
- building b400 : F1 building = **0.8504**, IoU building = **0.7398**.

## Modèle d'affaires envisagé

- SaaS institutionnel pour collectivités, assureurs et organismes de crise;
- facturation par crise ou par volume d'analyse;
- accompagnement et intégration sur mesure pour workflows SIG.

## Limites

- prototype académique, non opérationnel terrain;
- dépendance à la qualité et disponibilité des images satellite;
- erreurs possibles du modèle;
- classes simplifiées à 3 classes;
- supervision humaine nécessaire.

## Perspectives

- retour aux 5 classes xView2;
- robustesse accrue selon catastrophes et zones géographiques;
- intégration SIG future;
- exports GeoJSON / GeoTIFF futurs;
- intégration QGIS / ArcGIS future.
