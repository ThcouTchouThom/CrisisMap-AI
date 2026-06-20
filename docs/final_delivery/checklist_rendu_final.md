# Checklist du rendu final

## Code et dépôt

- [ ] `README.md` racine à jour pour un évaluateur.
- [ ] Application classique présente : `app/streamlit_app.py`.
- [ ] Aucun entraînement relancé pendant la préparation du rendu.
- [ ] Aucun checkpoint ou dataset complet ajouté par erreur.
- [ ] `.gitignore` protège les caches, outputs, checkpoints et archives locales.
- [ ] Checkpoint damage portable présent et versionnable.
- [ ] Checkpoint building portable présent et versionnable.
- [ ] Exemples embarqués présents dans `sample_data/demo_pairs/`.

## Livrables

- [ ] Rapport final 10 à 15 pages prêt.
- [ ] Pitch deck final prêt.
- [ ] Vidéo promotionnelle 2 à 3 minutes publiée ou prête.
- [ ] Lien YouTube renseigné dans `docs/final_delivery/video_demo_youtube_link.txt`.
- [ ] Fiche produit une page prête : `docs/final_delivery/fiche_produit_1page.md`.

## Prototype

- [ ] Checkpoint damage disponible localement.
- [ ] Checkpoint building disponible localement.
- [ ] Application classique lancée avec succès.
- [ ] Application classique testée avec les exemples embarqués.
- [ ] Test upload manuel réalisé.
- [ ] Test dataset réalisé si les données xBD locales sont disponibles.
- [ ] Exports PNG/JSON vérifiés.

## Honnêteté produit

- [ ] Ne pas dire que GeoJSON existe déjà.
- [ ] Ne pas dire que GeoTIFF existe déjà.
- [ ] Ne pas dire que QGIS/ArcGIS est intégré.
- [ ] Ne pas dire qu'une API publique est disponible.
- [ ] Présenter SIG, GeoJSON, GeoTIFF et API comme perspectives.

## Vérifications finales

```powershell
python -m py_compile app\streamlit_app.py
git diff --check
git status
git diff --stat
```

## Préparation Git

Ne pas utiliser `git add .`.

Ajouter explicitement les fichiers validés, par exemple :

```powershell
git add README.md .gitignore docs/final_delivery
git add app/streamlit_app.py sample_data
```

Adapter la commande selon les fichiers réellement retenus.
