# Plan — segmentation binaire des bâtiments

## Pourquoi ajouter une segmentation bâtiment seule

Le modèle actuel d'Aftermath prédit directement trois classes :

- `0` : fond;
- `1` : bâtiment non endommagé;
- `2` : bâtiment endommagé.

L'expérience oracle avec masque bâtiment parfait a montré un gain important :

- IoU damaged : `0.4175` vers `0.5383`;
- F1 damaged : `0.5891` vers `0.6999`.

Cela indique qu'une partie des erreurs du modèle vient du fait qu'il doit
résoudre en même temps la localisation des bâtiments et l'estimation des
dommages. Une segmentation binaire dédiée peut donc devenir la première étape
d'un pipeline plus robuste.

## Nouvelle formulation

La tâche bâtiment est binaire :

- `0` : fond;
- `1` : bâtiment.

Le masque est dérivé des cibles xBD existantes :

```text
building = original_target > 0
```

Le comportement des modes `3-class` et `5-class` reste inchangé pour la pipeline
de dommage.

## Pourquoi commencer avec l'image pré-catastrophe

Le premier run recommandé utilise `--input-mode pre`.

Raisons :

- l'image pré-catastrophe montre souvent les bâtiments avant destruction ou
  obstruction;
- la tâche est de localiser les bâtiments, pas encore de comparer les dommages;
- cela réduit le nombre de canaux d'entrée de 6 à 3 et simplifie le premier
  entraînement;
- le résultat pourra ensuite servir de masque structurel pour le post-traitement
  des prédictions de dommage.

Les modes `post` et `pre-post` restent disponibles pour comparer les variantes.

## Split recommandé

Le premier entraînement sérieux devrait utiliser un grand split no-leak, par
exemple :

```text
data/processed/splits_noleak_full_train
```

Ce split maximise la diversité tout en excluant les paires présentes dans la
validation et le test communs. La validation et le test ne doivent jamais être
augmentés.

## Modèles

Le script supporte :

- `unet` : U-Net local déjà présent dans le projet;
- `unetplusplus_effb3` : U-Net++ EfficientNet-B3 si
  `segmentation_models_pytorch` est installé.

Si `segmentation_models_pytorch` n'est pas disponible, le script bascule
proprement vers le U-Net local et l'indique dans la console et les checkpoints.

## Pertes

Pertes disponibles :

- `bce-dice`;
- `dice-bce`;
- `focal-tversky`.

La perte `focal-tversky` est la valeur par défaut. Les paramètres par défaut
sont :

```text
alpha = 0.3
beta = 0.7
gamma = 0.75
```

Ce réglage pénalise davantage les faux négatifs et favorise légèrement le rappel
des bâtiments.

## Métriques

Les métriques suivies sont :

- pixel accuracy;
- mean IoU;
- IoU fond;
- IoU bâtiment;
- précision bâtiment;
- rappel bâtiment;
- F1 bâtiment.

Le checkpoint `best_building.pt` est sélectionné selon l'IoU bâtiment en
validation.

## Commande recommandée locale

Exemple pour une première nuit d'entraînement sur GPU Windows :

```powershell
python scripts/train_building_segmentation.py `
  --root data/raw/xbd/train `
  --train-csv data/processed/splits_noleak_full_train/train_pairs.csv `
  --val-csv data/processed/splits_noleak_full_train/val_pairs.csv `
  --test-csv data/processed/splits_noleak_full_train/test_pairs.csv `
  --output-dir outputs/checkpoints/building_pre_unetplusplus_effb3_1024 `
  --model unetplusplus_effb3 `
  --input-mode pre `
  --image-size 1024 `
  --batch-size 2 `
  --epochs 50 `
  --lr 1e-4 `
  --loss focal-tversky `
  --augment-mode safe `
  --target-mode building-binary `
  --device cuda `
  --amp `
  --num-workers 0
```

Pour un smoke test CPU très court :

```powershell
python scripts/train_building_segmentation.py `
  --root data/raw/xbd/train `
  --train-csv data/processed/splits_noleak_full_train/train_pairs.csv `
  --val-csv data/processed/splits_noleak_full_train/val_pairs.csv `
  --output-dir outputs/checkpoints/building_smoke `
  --model unet `
  --input-mode pre `
  --image-size 256 `
  --batch-size 1 `
  --epochs 1 `
  --loss bce-dice `
  --augment-mode none `
  --target-mode building-binary `
  --device cpu `
  --num-workers 0 `
  --max-train-samples 4 `
  --max-val-samples 2
```

## Lien avec la suite du projet

Si cette segmentation bâtiment fonctionne bien, elle pourra alimenter une
pipeline en deux étapes :

1. segmenter les bâtiments;
2. classifier chaque bâtiment, ou chaque pixel bâtiment, en intact/endommage.

Elle permettra aussi de tester un post-traitement de type composante majoritaire
sur les prédictions de dommage, inspiré de l'expérience oracle.
