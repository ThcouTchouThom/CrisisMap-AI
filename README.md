# Aftermath / CrisisMap AI

**Voir les dégâts pour agir plus vite.**

Aftermath est un prototype IA de cartographie automatique des dommages à partir de paires d'images satellite **avant / après catastrophe**. Le projet vise à aider une cellule de crise, une ONG, une collectivité ou un analyste SIG à obtenir rapidement une première lecture visuelle des bâtiments intacts et endommagés.

Le dépôt contient le code source, les scripts d'expérimentation, les applications Streamlit de démonstration et les documents de rendu final. Les données xBD/xView2 complètes, les sorties expérimentales lourdes et les checkpoints complets restent hors Git.

## Objectif du projet

L'objectif est de transformer une paire satellite pré/post catastrophe en une carte de segmentation à 3 classes :

| Classe | Signification |
| --- | --- |
| `0` | fond / absence de bâtiment |
| `1` | bâtiment intact ou non endommagé |
| `2` | bâtiment endommagé |

Le prototype actuel démontre une chaîne complète :

```text
image pré + image post
-> modèle damage Siamese Attention
-> TTA d4
-> segmentation bâtiment U-Net++ EfficientNet-B4
-> post-processing par composantes bâtiment
-> overlay final + masques + incertitude + exports PNG/JSON
```

## Fonctionnalités du prototype

Les applications Streamlit permettent :

- de sélectionner une paire du dataset xBD si les données sont présentes localement;
- de téléverser manuellement une image avant et une image après catastrophe;
- de lancer une inférence damage;
- d'utiliser la TTA d4 pour stabiliser la prédiction;
- d'utiliser un masque bâtiment prédit;
- d'appliquer un post-processing par `component majority`;
- d'afficher l'overlay final sur l'image post-catastrophe;
- d'afficher les sorties intermédiaires : damage brut, masque bâtiment, damage final;
- d'afficher l'incertitude;
- d'afficher les métriques quand une vérité terrain est disponible;
- d'exporter des PNG et un rapport JSON.

Le prototype **n'exporte pas encore** de GeoJSON, GeoTIFF ou projet SIG complet. L'intégration QGIS/ArcGIS, une API publique et le géoréférencement opérationnel sont des perspectives futures.

## Architecture globale du dépôt

```text
app/
  streamlit_app.py              # Application classique stable.

src/crisismap/
  data/                         # Dataset xBD, indexation et préparation.
  evaluation/                   # Évaluation et visualisation de prédictions.
  models/                       # Modèles U-Net, Siamese, multi-temporal, etc.
  training/                     # Entraînement du baseline damage.

scripts/                        # Évaluations, campagnes, outils de résumé.
configs/                        # Configurations de campagnes expérimentales.
slurm/                          # Scripts Rorqual / Alliance.
docs/                           # Documentation, plans, sources NotebookLM.
docs/final_delivery/            # Livrables finaux et stratégie de rendu.
sample_data/                    # Petites paires embarquées pour tester sans xBD complet.
data/                           # Données locales ignorées par Git.
outputs/                        # Sorties ignorées, sauf les deux checkpoints portables retenus.
demo_assets/                    # Exemples locaux de démo, ignorés par Git.
```

## Prérequis

- Python 3.11 recommandé;
- Windows PowerShell ou Linux/macOS shell;
- GPU CUDA recommandé pour une démonstration fluide, mais le CPU reste possible pour des tests légers;
- accès local aux checkpoints nécessaires;
- optionnel : données xBD/xView2 pour le mode dataset.

## Installation locale Windows

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Installation locale Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Dépendances principales

Le fichier `requirements.txt` contient notamment :

- `torch`, `torchvision`;
- `streamlit`;
- `segmentation-models-pytorch==0.5.0`;
- `timm==1.0.27`;
- `numpy`, `pandas`, `pillow`, `opencv-python`, `matplotlib`;
- `shapely` pour les annotations xBD.

## Checkpoints inclus dans le dépôt final

Le dépôt final inclut les deux checkpoints portables nécessaires à l'application :

```text
outputs/checkpoints/dftv2_hist1000_attention_sqrt2_ft_250_seed0/best_damage_arch_portable.pt
outputs/checkpoints/b400_effb4_sampler8_ft/best_building_portable.pt
```

Fallbacks acceptés par l'application :

```text
outputs/checkpoints/dftv2_hist1000_attention_sqrt2_ft_250_seed0/best_damage_arch.pt
outputs/checkpoints/b400_effb4_sampler8_ft/best_building.pt
```

Tailles locales constatées :

- damage portable : environ 33.9 Mo;
- building portable : environ 80.2 Mo.

Ces deux fichiers sont sous la limite GitHub de 100 Mo par fichier. Les autres checkpoints expérimentaux restent exclus.

## Données nécessaires

### Mode upload manuel

Le mode upload fonctionne sans dataset complet : il suffit de fournir deux images RGB compatibles, avant et après catastrophe.

### Mode exemples dataset

Pour utiliser les exemples xBD dans l'application, les données doivent être présentes localement :

```text
data/raw/xbd/train/images/
data/raw/xbd/train/labels/
data/raw/xbd/train/targets/
data/processed/splits/
```

Les données complètes xBD/xView2 ne sont pas incluses dans Git.

## Lancer l'application classique

```powershell
python -m streamlit run app/streamlit_app.py
```

Cette version est la version sûre pour l'évaluation.

## Tester rapidement la compilation

```powershell
python -m py_compile app/streamlit_app.py
```

## Tester avec les exemples embarqués

Le dépôt final contient quelques paires légères dans :

```text
sample_data/demo_pairs/
```

Pour les utiliser :

1. lancer l'application classique;
2. choisir le mode **Exemples inclus**;
3. sélectionner une paire;
4. cliquer sur **Analyser**.

Ce mode ne nécessite pas le dataset xBD complet.

## Tester avec les exemples dataset

1. Vérifier que les dossiers `data/raw/xbd/train/` et `data/processed/splits/` existent.
2. Lancer l'application.
3. Choisir le mode dataset.
4. Sélectionner une paire recommandée.
5. Lancer l'inférence.

Les paires recommandées sont définies dans `RECOMMENDED_PAIR_IDS` dans `app/streamlit_app.py`.

## Tester avec upload manuel

1. Lancer l'application.
2. Choisir le mode upload.
3. Charger une image avant catastrophe.
4. Charger une image après catastrophe.
5. Lancer l'inférence.

Sans vérité terrain, l'application affiche une prédiction et des statistiques, mais pas de métriques supervisées.

## Exports disponibles

Exports actuels :

- masque PNG;
- overlay PNG;
- rapport JSON.

Exports non disponibles dans le prototype actuel :

- GeoJSON;
- GeoTIFF;
- projet QGIS/ArcGIS;
- API publique;
- géoréférencement opérationnel.

Ces éléments sont documentés comme perspectives futures.

## Résultats principaux

| Modèle / pipeline | F1 damaged | IoU damaged | Mean IoU |
| --- | ---: | ---: | ---: |
| Baseline U-Net + TTA d4 | 0.6313 | 0.4612 | 0.6816 |
| Ancien champion Siamese | 0.6788 | 0.5138 | 0.7073 |
| Champion intégré `dftv2_hist1000_attention_sqrt2_ft_250_seed0` | 0.7013 | 0.5400 | 0.7283 |
| Dernier run marginalement meilleur `dftv2_hist1000_attention_sqrt4_ft_400_seed0` | 0.7018 | 0.5406 | 0.7273 |

Décision finale : le modèle intégré reste `dftv2_hist1000_attention_sqrt2_ft_250_seed0`, car il est stabilisé, portable et testé dans l'application. Le dernier run disponible est seulement marginalement meilleur.

Champion building :

| Modèle | F1 building | IoU building |
| --- | ---: | ---: |
| `b400_effb4_sampler8_ft` - U-Net++ EfficientNet-B4 | 0.8504 | 0.7398 |

## Limites connues

- prototype académique, non opérationnel terrain;
- dépendance forte à la disponibilité et à la qualité des images satellite;
- risque d'erreur sur les bâtiments et les dommages;
- formulation actuelle simplifiée à 3 classes;
- pas encore de score officiel xView2 5 classes;
- pas encore d'intégration SIG complète;
- supervision humaine nécessaire.

Aftermath doit être présenté comme une aide à la décision, pas comme un outil remplaçant l'expertise humaine.

## Livrables finaux

Les fichiers de préparation du rendu final sont dans :

```text
docs/final_delivery/
```

Contenu attendu :

- `README_LIVRABLES_FINAUX.md`;
- `fiche_produit_1page.md`;
- `video_demo_youtube_link.txt`;
- `checklist_rendu_final.md`;
- `repo_cleanup_report.md`;
- `checkpoints_and_data_strategy.md`.

Le pitch deck final, le rapport final 10-15 pages et la vidéo de démonstration peuvent être référencés depuis ce dossier.

## Équipe

- Thomas GOURJAULT;
- Grégory JOURDAIN;
- Aurélien CASAGRANDI;
- Matthis LAHARGOUE.

## Crédit dataset

Le projet utilise xBD / xView2 comme dataset de référence pour l'évaluation des dommages sur bâtiments à partir d'images satellite pré/post catastrophe.

À citer dans le rapport ou les slides :

- xBD: A Dataset for Assessing Building Damage from Satellite Imagery;
- xView2 Challenge: Assess Building Damage.
