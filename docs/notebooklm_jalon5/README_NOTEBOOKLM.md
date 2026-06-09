# Sources NotebookLM - Jalon 5

Ce dossier sert uniquement de source documentaire pour NotebookLM afin de générer une présentation orale claire, structurée et convaincante du projet **Aftermath / CrisisMap AI**.

Ce dossier n'est pas le rapport final rendu au professeur. Il contient des notes préparatoires, des messages clés, des résultats, un script oral et un pitch.

## Fichiers

| Fichier | Rôle |
| --- | --- |
| `00_resume_executif.md` | Vue d'ensemble du projet, résultats clés et message principal |
| `01_probleme_et_solution.md` | Narration problème réel, utilisateurs et solution |
| `02_pipeline_technique.md` | Description pédagogique du pipeline IA |
| `03_resultats_modeles.md` | Résultats quantitatifs et comparaison des modèles |
| `04_application_demo.md` | Description de l'application Streamlit et du déroulé de démonstration |
| `05_limites_et_roadmap.md` | Limites actuelles, risques et prochaines étapes |
| `06_script_presentation_10min.md` | Proposition de script oral slide par slide |
| `07_pitch_1min.md` | Pitch court pour présenter le projet rapidement |

## Ordre conseillé pour NotebookLM

Pour générer une présentation, utiliser les fichiers dans cet ordre :

1. `00_resume_executif.md`
2. `01_probleme_et_solution.md`
3. `02_pipeline_technique.md`
4. `03_resultats_modeles.md`
5. `04_application_demo.md`
6. `05_limites_et_roadmap.md`
7. `06_script_presentation_10min.md`
8. `07_pitch_1min.md`

## Message central à faire ressortir

Aftermath est passé d'une baseline de segmentation fonctionnelle à un pipeline plus mature :

- modèle damage Siamese Attention performant ;
- TTA d4 ;
- modèle building U-Net++ EfficientNet-B4 ;
- post-processing par composante bâtiment ;
- application Streamlit démontrable ;
- visualisation explicable des prédictions.

## Résultats à citer

| Modèle | F1 damaged | IoU damaged | Mean IoU |
| --- | ---: | ---: | ---: |
| U-Net + TTA d4 | 0.631300 | 0.461240 | 0.681574 |
| Ancien champion Siamese | 0.678801 | 0.513776 | 0.707285 |
| Nouveau champion Siamese | 0.701317 | 0.540022 | 0.728266 |

Champion building :

| Modèle | F1 building | IoU building |
| --- | ---: | ---: |
| b400 EffB4 | 0.850421 | 0.739767 |

## Consigne pour NotebookLM

Générer une présentation orale pédagogique, orientée démonstration, avec une narration claire :

problème réel -> dataset -> baseline -> amélioration modèle -> segmentation bâtiment -> prototype -> impact -> limites -> roadmap.
