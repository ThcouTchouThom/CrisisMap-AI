# Prototype Streamlit

## Objectif

Le prototype Streamlit présente une interface simple pour visualiser :

- l'image avant catastrophe ;
- l'image après catastrophe ;
- le masque vérité terrain ;
- la prédiction du modèle ;
- une superposition de la prédiction sur l'image après catastrophe.

Fichier principal :

```text
app/streamlit_app.py
```

## Fonctionnalités

L'application permet :

- de choisir un split ;
- de choisir un `pair_id` ;
- de charger un checkpoint U-Net ;
- d'exécuter une inférence ;
- d'afficher quelques métriques simples sur la prédiction.

La légende utilise les classes :

- fond / absence de bâtiment ;
- bâtiment non endommagé ;
- bâtiment endommagé.

## État

Le prototype est fonctionnel. Il devra être mis à jour avec le meilleur modèle final no-leak lorsque la campagne augmentation/sampler sera terminée.

Pour le jalon 2, il démontre la direction applicative : passer d'un modèle de segmentation à une interface utilisable pour explorer rapidement des dommages post-catastrophe.

