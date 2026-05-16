# Fuite de données et protocole no-leak

## Problème découvert

Des comparaisons initiales utilisaient `data/processed/splits_full/test_pairs.csv` comme test global commun. Plus tard, un chevauchement a été trouvé : certains `pair_id` du test global apparaissaient dans le train ou la validation de splits alternatifs.

Conséquence : certaines métriques étaient probablement gonflées.

## Correction

Le protocole corrigé fixe deux fichiers communs :

- validation commune : `data/processed/splits_full/val_pairs.csv` ;
- test commun : `data/processed/splits_full/test_pairs.csv`.

Tous les splits d'entraînement ultérieurs doivent :

- copier exactement ces fichiers pour val/test ;
- exclure tous leurs `pair_id` du train ;
- appliquer augmentation et sampling uniquement au train.

## Importance pour le jalon

Cette correction montre une maturité méthodologique importante. Le projet ne se limite pas à produire un score : il vérifie que les scores sont comparables et non contaminés.

Pour la présentation, il faut rester concis :

- mentionner qu'une fuite potentielle a été identifiée ;
- expliquer la correction common_val/common_test ;
- montrer que les résultats propres no-leak sont maintenant la référence.

