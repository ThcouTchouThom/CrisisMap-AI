# Livrables finaux - Aftermath / CrisisMap AI

Ce dossier centralise les éléments de rendu final du projet **Aftermath**, prototype IA de cartographie automatique des dommages à partir d'images satellite avant/après catastrophe.

## Contenu du rendu final attendu

| Élément | Statut | Emplacement conseillé |
| --- | --- | --- |
| Code source organisé | présent | dépôt Git |
| Prototype Streamlit classique | présent | `app/streamlit_app.py` |
| Rapport final 10 à 15 pages | à joindre | `docs/final_delivery/` ou document séparé |
| Pitch deck final | à joindre | `docs/final_delivery/` ou support externe |
| Vidéo promotionnelle 2 à 3 minutes | lien à renseigner | `docs/final_delivery/video_demo_youtube_link.txt` |
| Fiche produit une page | présente | `docs/final_delivery/fiche_produit_1page.md` |
| README technique d'évaluation | présent | `README.md` |

## Prototype démontrable

Commande recommandée :

```powershell
python -m streamlit run app/streamlit_app.py
```

Le prototype actuel permet l'upload, l'utilisation d'exemples embarqués ou la sélection dataset d'une paire satellite avant/après. Il lance l'inférence damage, la TTA d4, la segmentation bâtiment, le post-processing par composantes, l'overlay final, les masques intermédiaires, l'incertitude et les exports PNG/JSON.

## Résultats clés

| Élément | Résultat |
| --- | --- |
| Baseline U-Net + TTA d4 | F1 damaged = 0.6313 |
| Champion intégré | F1 damaged = 0.7013, IoU damaged = 0.5400, mean IoU = 0.7283 |
| Building b400 | F1 building = 0.8504, IoU building = 0.7398 |

## Points à ne pas sur-vendre

Le prototype n'exporte pas encore :

- GeoJSON;
- GeoTIFF;
- projet QGIS/ArcGIS;
- API publique;
- géoréférencement opérationnel.

Ces éléments sont des perspectives futures et doivent être présentés comme tels.

## Fichiers de ce dossier

- `fiche_produit_1page.md` : synthèse produit courte.
- `video_demo_youtube_link.txt` : emplacement du lien vidéo YouTube.
- `checklist_rendu_final.md` : checklist avant dépôt.
- `repo_cleanup_report.md` : audit de nettoyage conservateur.
- `checkpoints_and_data_strategy.md` : stratégie pour les poids et données.

## Commandes de vérification

```powershell
python -m py_compile app\streamlit_app.py
git diff --check
git status
git diff --stat
```
