# Note de rendu - Jalon 2

## Projet et équipe

**Aftermath / CrisisMap AI**

Équipe : Thomas Gourjault, Aurélien Casagrandi, Grégory Jourdain.

## Objectif du jalon

Le projet vise la segmentation des dommages aux bâtiments après catastrophe à partir de paires d'images satellites avant/après. Pour le jalon 2, le rendu montre que :

- les données sont accessibles, chargées et visualisées ;
- un pipeline train/validation/test existe ;
- une première exploration du dataset est réalisée ;
- un baseline d'IA est entraîné et évalué ;
- le dépôt Git est structuré pour poursuivre le prototype.

## Contenu du rendu

Le dossier `deliverables/jalon_2/` contient :

- la présente note de synthèse ;
- le dossier `presentation/` pour le PDF de présentation orale ;
- des sources Markdown dans `notebooklm_sources/` ;
- une synthèse de résultats dans `results/` ;
- quelques figures légères et des références de figures dans `figures/` ;
- un plan de slides dans `slides_draft/`.

## Données et formulation actuelle

Le dataset utilisé est **xBD/xView2**. Le jeu brut extrait contient **2 799 paires** d'images satellites pré/post catastrophe.

La cible actuelle est une simplification temporaire à 3 classes :

| Classe | Signification |
| --- | --- |
| 0 | background / absence de bâtiment |
| 1 | bâtiment non endommagé |
| 2 | bâtiment endommagé |

Cette formulation permet de valider le pipeline rapidement. Un objectif futur est de revenir vers une segmentation multi-niveaux des dommages, plus proche des classes originales du dataset.

Les splits peuvent filtrer les paires avec trop peu d'information bâtiment, notamment avec `min_nonzero_ratio >= 0.01`.

## Structure utile du dépôt

```text
app/                         # Prototype Streamlit.
data/                        # Données locales et CSV traités.
scripts/                     # Setup, génération de splits et utilitaires.
slurm/                       # Préparation Alliance / Rorqual.
src/crisismap/data/          # Inspection, indexation, Dataset PyTorch.
src/crisismap/models/        # U-Net baseline.
src/crisismap/training/      # Entraînement.
src/crisismap/evaluation/    # Évaluation et prédictions.
deliverables/jalon_2/        # Rendu de ce jalon.
```

## Pipeline de données

Le pipeline suit les étapes suivantes :

1. extraction des archives xBD/xView2 ;
2. inspection des dossiers `images`, `labels` et `targets` ;
3. visualisation des paires avant/après et des masques ;
4. construction de `data/processed/xbd_train_index.csv` ;
5. création de splits train/validation/test ;
6. chargement des paires par un Dataset PyTorch.

## Exploration initiale

L'exploration met en évidence :

- une distribution par catastrophe ;
- une domination du background ;
- une classe `damaged` rare et difficile ;
- l'intérêt de `nonzero_ratio` pour mesurer l'information bâtiment ;
- l'intérêt de `damage_ratio` pour suivre les pixels endommagés.

La pixel accuracy seule est insuffisante dans ce contexte. L'IoU damaged et le F1 damaged sont plus proches de l'objectif applicatif.

## Baseline IA

Le baseline est un **U-Net** de segmentation sémantique.

- entrée : 6 canaux, soit pré RGB + post RGB ;
- sortie : masque 3 classes ;
- perte de référence : **CE-Dice** ;
- pondération des classes : `[0.05, 1.0, 4.0]`.

Des expériences 512 et 1024 pixels ont été préparées et exécutées pour valider puis étendre ce baseline.

## Résumé des métriques

Le premier baseline fonctionnel obtient environ :

| Référence | Mean IoU | IoU damaged |
| --- | ---: | ---: |
| Baseline initial | ~0.62 | ~0.38 |

Le protocole propre no-leak actuel donne une référence plus récente :

| Référence | Mean IoU | IoU damaged |
| --- | ---: | ---: |
| Référence no-leak | ~0.66 | ~0.416 |

Ces résultats montrent que la chaîne fonctionne et que la classe damaged reste le point de progression principal.

## Correction méthodologique

Une fuite de données a été identifiée dans certaines comparaisons initiales : des paires utilisées pour un test global pouvaient apparaître dans des splits d'entraînement alternatifs.

Le protocole a été corrigé :

- validation commune : `data/processed/splits_full/val_pairs.csv` ;
- test commun : `data/processed/splits_full/test_pairs.csv` ;
- les nouveaux trains excluent tous les `pair_id` de ces deux fichiers.

Cette correction no-leak est maintenant la référence méthodologique.

## Prototype et passage à l'échelle

Un premier prototype Streamlit fonctionnel existe. Il permet de visualiser les images d'entrée, les masques et les prédictions sur un échantillon.

La préparation Rorqual / SLURM existe aussi pour passer à des entraînements plus lourds, notamment en 1024 pixels sur GPU H100.

## Reproduire les étapes principales

Depuis la racine du dépôt, après avoir placé les archives xBD :

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_project.ps1
python src/crisismap/data/inspect_xbd.py --root data/raw/xbd/train
python src/crisismap/visualization/visualize_xbd_sample.py --root data/raw/xbd/train --mode 3-class
```

Entraîner puis évaluer un baseline court :

```powershell
python src/crisismap/training/train_unet.py `
  --root data/raw/xbd/train `
  --train-csv data/processed/splits/train_pairs.csv `
  --val-csv data/processed/splits/val_pairs.csv `
  --output-dir outputs/checkpoints/unet_baseline_jalon2 `
  --image-size 512 `
  --batch-size 2 `
  --epochs 5 `
  --target-mode 3-class `
  --loss ce-dice `
  --class-weights 0.05 1.0 4.0

python src/crisismap/evaluation/evaluate_unet.py `
  --root data/raw/xbd/train `
  --split-csv data/processed/splits/test_pairs.csv `
  --checkpoint outputs/checkpoints/unet_baseline_jalon2/best_unet.pt `
  --output outputs/predictions/unet_baseline_jalon2_test_metrics.json `
  --image-size 512 `
  --target-mode 3-class
```

Prototype :

```powershell
streamlit run app/streamlit_app.py
```

## Limites et prochaines étapes

Limites actuelles :

- formulation 3 classes encore simplifiée ;
- classe damaged fortement déséquilibrée ;
- prototype cartographique encore partiel ;
- résultats avancés en cours de consolidation.

Prochaines étapes :

- finaliser les campagnes augmentation/sampler no-leak ;
- mettre à jour Streamlit avec le meilleur checkpoint propre ;
- analyser les faux positifs et faux négatifs damaged ;
- tester des architectures plus fortes ;
- exploiter les métadonnées géospatiales pour une visualisation de carte.
