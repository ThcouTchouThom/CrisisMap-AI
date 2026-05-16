# Pipeline de données

## Source

Le dataset utilisé est xBD/xView2. Il contient des paires d'images satellites avant/après catastrophe, des annotations JSON et des masques cibles.

Archives attendues localement :

- `data/raw/archives/train_images_labels_targets.tar`
- `data/raw/archives/xview_geotransforms.json.tgz`

Structure attendue après extraction :

```text
data/raw/xbd/train/images/
data/raw/xbd/train/labels/
data/raw/xbd/train/targets/
data/raw/geotransforms/xview_geotransforms.json
```

Les données brutes ne sont pas suivies dans Git.

## Rôle des scripts

`inspect_xbd.py` vérifie que les dossiers `images`, `labels` et `targets` existent, compte les fichiers et vérifie les paires pré/post.

`build_xbd_index.py` crée une ligne par paire d'images. Il collecte :

- `pair_id` ;
- catastrophe ;
- chemins relatifs des images, labels et masques ;
- nombre de bâtiments annotés ;
- valeurs uniques dans le masque ;
- ratio de pixels bâtiment (`nonzero_ratio`) ;
- ratio de pixels endommagés (`damage_ratio`, calculé à partir des classes 2, 3, 4).

`create_xbd_splits.py` et les scripts avancés créent les CSV train/validation/test.

## Formulation utilisée au jalon 2

La cible est simplifiée en 3 classes :

- 0 : arrière-plan ;
- 1 : bâtiment non endommagé ;
- 2 : bâtiment endommagé.

Les classes originales de dommages xBD peuvent être réintroduites plus tard pour une segmentation multi-niveaux.

## Géoréférencement

`xview_geotransforms.json.tgz` contient des informations de géoréférencement. Le fichier extrait `xview_geotransforms.json` pourrait servir à replacer les prédictions dans un système géographique ou dans une carte interactive.

À ce stade, ces métadonnées sont extraites mais pas encore utilisées par le dataset PyTorch ni par l'entraînement.

