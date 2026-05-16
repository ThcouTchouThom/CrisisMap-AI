# Exploration du dataset

## Volume

Le dataset brut extrait contient 2799 paires pré/post. Certains splits contiennent moins de paires, car un filtrage est appliqué pour enlever les images avec trop peu d'information utile sur les bâtiments.

Filtre important :

```text
min_nonzero_ratio >= 0.01
```

Ce filtre retire des images où les pixels de bâtiments sont trop rares.

## Split initial old4

Le premier split utilisé quatre catastrophes :

- `hurricane-harvey` ;
- `hurricane-michael` ;
- `santa-rosa-wildfire` ;
- `palu-tsunami`.

Raisons :

- commencer avec un sous-ensemble gérable ;
- représenter différents types de catastrophes ;
- réduire le temps d'entraînement pendant le débogage ;
- valider toute la chaîne avant de passer à plus grande échelle ;
- choisir des catastrophes avec des exemples de dommages significatifs.

## Full split

Le split `splits_full` utilise les 10 catastrophes disponibles après filtrage. Il a été créé avec un seed fixe et des ratios standards train/validation/test.

Les fichiers importants sont :

- `data/processed/splits_full/val_pairs.csv` ;
- `data/processed/splits_full/test_pairs.csv`.

Ces fichiers sont ensuite devenus la validation commune et le test commun pour les protocoles no-leak.

## Difficultés observées

La classe `damaged` est rare comparée au background et aux bâtiments non endommagés. Cela crée un déséquilibre fort :

- une bonne accuracy pixel peut être obtenue en prédisant surtout le background ;
- l'IoU de la classe damaged est beaucoup plus représentatif de l'objectif réel ;
- le F1 damaged permet de suivre le compromis précision/rappel.

Deux ratios sont utiles :

- `nonzero_ratio` : proportion de pixels bâtiment ou dommage dans le masque ;
- `damage_ratio` : proportion de pixels appartenant aux classes originales de dommage 2, 3 et 4.

