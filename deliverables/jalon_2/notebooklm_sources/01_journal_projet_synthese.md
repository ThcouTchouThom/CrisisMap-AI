# Synthèse du journal de projet

## Projet

Nom : CrisisMap AI / Aftermath.

Objectif : détecter et cartographier automatiquement les dommages aux bâtiments après une catastrophe naturelle à partir d'images satellites avant/après.

Dataset : xBD/xView2, avec images pré-catastrophe, images post-catastrophe, annotations JSON et masques de segmentation.

## Avancement principal

Le pipeline complet a été mis en place :

1. extraction locale du dataset ;
2. inspection de la structure ;
3. visualisation des paires d'images et des masques ;
4. construction d'un index CSV ;
5. création de splits train/validation/test ;
6. implémentation d'un dataset PyTorch ;
7. implémentation d'un U-Net léger ;
8. entraînement baseline ;
9. évaluation quantitative ;
10. visualisation des prédictions ;
11. prototype Streamlit ;
12. préparation de scripts SLURM pour Rorqual.

## Résultat méthodologique important

Une fuite de données a été identifiée dans une première stratégie de comparaison. Certains `pair_id` du test global apparaissaient dans le train ou la validation de splits alternatifs. Le protocole a été corrigé :

- `data/processed/splits_full/val_pairs.csv` devient la validation commune ;
- `data/processed/splits_full/test_pairs.csv` devient le test commun ;
- les nouveaux splits d'entraînement excluent tous les `pair_id` de ces deux fichiers.

Cette correction est un point fort du projet : elle évite de surestimer les performances.

## Résultat technique

Le meilleur baseline propre actuel avant la campagne augmentation/sampler est approximativement :

- Mean IoU : 0.658 à 0.665 ;
- IoU damaged : 0.416 à 0.418 ;
- F1 damaged : 0.587 à 0.589.

Ces valeurs dépassent le niveau strictement attendu pour le jalon 2, mais elles doivent être présentées comme des résultats de recherche en cours.

