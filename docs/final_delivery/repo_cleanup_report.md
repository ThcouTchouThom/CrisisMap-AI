# Rapport de nettoyage conservateur du dépôt

Ce rapport décrit l'état du dépôt et les décisions de nettoyage recommandées pour le rendu final. Aucun fichier risqué n'a été supprimé pendant cette préparation.

## État observé

La commande `git status --short` montre :

- `app/streamlit_app.py` modifié;
- plusieurs archives ZIP de jalons non suivies;
- `campaign2_aug_sampler_export.tgz` et son dossier extrait;
- `demo_assets/`;
- anciennes sources NotebookLM désormais supprimées au profit de `docs/final_delivery/`;
- journaux complets locaux;
- fichiers `souvenir` / `souvenir2`;
- `video_demo_youtube_link.txt` à la racine;
- scripts et configs expérimentaux non suivis.

## Ce qui doit être versionné

À versionner pour le rendu final :

- code source stable :
  - `app/`;
  - `src/`;
  - `scripts/` utiles;
  - `configs/` utiles;
  - `slurm/` si les scripts d'expérimentation doivent rester documentés;
- documentation :
  - `README.md`;
  - `docs/`;
  - `docs/final_delivery/`;
- exemples embarqués légers :
  - `sample_data/demo_pairs/`;
- fichiers de configuration légers :
  - `requirements.txt`;
  - `.gitignore`;
  - `.gitattributes`.

## Ce qui doit rester local mais ignoré

À garder localement mais ne pas pousser :

- `outputs/`;
- `data/raw/xbd/`;
- `data/processed/` complet;
- `demo_assets/`;
- `*.pt`, `*.pth`, `*.ckpt`;
- archives `.zip`, `.tgz`, `.tar`, `.tar.gz`;
- logs;
- `campaign2_aug_sampler_export/`;
- journaux complets de travail;
- fichiers `souvenir*`;
- environnements virtuels.

## Ce qui peut être supprimé après validation humaine

Ne pas supprimer automatiquement. Après validation, les éléments suivants peuvent être retirés localement si l'espace disque manque :

- `campaign2_aug_sampler_export/`;
- `campaign2_aug_sampler_export.tgz`;
- anciennes archives ZIP de jalons déjà soumises;
- journaux complets locaux si leur contenu est résumé ailleurs;
- `souvenir` et `souvenir2`;
- exports locaux volumineux dans `demo_assets/`;
- checkpoints expérimentaux non retenus.

## Checkpoints retenus

Checkpoints nécessaires au prototype final :

```text
outputs/checkpoints/dftv2_hist1000_attention_sqrt2_ft_250_seed0/best_damage_arch_portable.pt
outputs/checkpoints/b400_effb4_sampler8_ft/best_building_portable.pt
```

Fallbacks :

```text
outputs/checkpoints/dftv2_hist1000_attention_sqrt2_ft_250_seed0/best_damage_arch.pt
outputs/checkpoints/b400_effb4_sampler8_ft/best_building.pt
```

Tailles locales constatées :

- damage : environ 33.9 Mo;
- building : environ 80.2 Mo.

## Données minimales nécessaires pour tester

### Test upload

Deux images RGB suffisent :

- image avant catastrophe;
- image après catastrophe.

### Test dataset

Requiert localement :

```text
data/raw/xbd/train/images/
data/raw/xbd/train/labels/
data/raw/xbd/train/targets/
data/processed/splits/
```

Le dataset complet ne doit pas être versionné.

## Mises à jour `.gitignore`

Ajouts recommandés et appliqués :

- dossiers de rendu locaux;
- archives `.zip`, `.tgz`, `.tar`, `.tar.gz`;
- `demo_assets/`;
- `campaign2_aug_sampler_export/`;
- journaux complets locaux;
- `souvenir*`;
- runner PowerShell local;
- lien vidéo local à la racine.

## Commandes recommandées de vérification

```powershell
python -m py_compile app\streamlit_app.py
git diff --check
git status
git diff --stat
```

## Commande `git add` recommandée

Ne pas utiliser `git add .`.

Proposition prudente :

```powershell
git add README.md .gitignore docs/final_delivery
git add app/streamlit_app.py sample_data
git add scripts/select_jalon5_demo_pairs.py
```

Ajouter séparément les scripts/configs expérimentaux uniquement s'ils doivent faire partie du rendu final.
