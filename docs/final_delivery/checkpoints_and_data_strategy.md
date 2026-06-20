# Stratégie checkpoints et données

Ce document décrit quels fichiers sont nécessaires pour tester le prototype Aftermath, lesquels doivent rester locaux et comment éviter de versionner des artefacts lourds.

## Checkpoints nécessaires au prototype

### Damage champion intégré

Chemin prioritaire :

```text
outputs/checkpoints/dftv2_hist1000_attention_sqrt2_ft_250_seed0/best_damage_arch_portable.pt
```

Fallback :

```text
outputs/checkpoints/dftv2_hist1000_attention_sqrt2_ft_250_seed0/best_damage_arch.pt
```

Taille locale constatée : environ **33.9 Mo**.

### Building champion

Chemin prioritaire :

```text
outputs/checkpoints/b400_effb4_sampler8_ft/best_building_portable.pt
```

Fallback :

```text
outputs/checkpoints/b400_effb4_sampler8_ft/best_building.pt
```

Taille locale constatée : environ **80.2 Mo**.

## Checkpoints versionnés dans le rendu final

Le rendu GitHub clé en main doit inclure uniquement ces deux checkpoints portables :

```text
outputs/checkpoints/dftv2_hist1000_attention_sqrt2_ft_250_seed0/best_damage_arch_portable.pt
outputs/checkpoints/b400_effb4_sampler8_ft/best_building_portable.pt
```

Ils sont sous la limite GitHub de 100 Mo par fichier. Les règles `.gitignore` autorisent explicitement ces deux fichiers tout en gardant le reste de `outputs/` ignoré.

## À ne pas versionner par défaut

Ne pas versionner :

- checkpoints expérimentaux complets;
- dossiers `outputs/`;
- dataset xBD complet;
- images brutes extraites;
- logs Rorqual;
- exports locaux de figures;
- archives de campagnes.

Les fichiers `.pt`, `.pth`, `.ckpt` et `outputs/*` sont ignorés par `.gitignore`, sauf les deux checkpoints portables listés ci-dessus.

## Si Git ignore encore les checkpoints

Selon l'ordre des règles `.gitignore`, Git peut encore nécessiter un ajout forcé. Dans ce cas :

```powershell
git add -f outputs\checkpoints\dftv2_hist1000_attention_sqrt2_ft_250_seed0\best_damage_arch_portable.pt
git add -f outputs\checkpoints\b400_effb4_sampler8_ft\best_building_portable.pt
```

## Données nécessaires

### Pour tester sans dataset complet

Le mode upload manuel suffit :

- image pré-catastrophe RGB;
- image post-catastrophe RGB.

L'application produit alors une prédiction et des exports PNG/JSON, mais pas de métriques supervisées.

### Pour tester avec exemples dataset

Structure attendue :

```text
data/raw/xbd/train/images/
data/raw/xbd/train/labels/
data/raw/xbd/train/targets/
data/processed/splits/test_pairs.csv
data/processed/splits/val_pairs.csv
data/processed/splits/train_pairs.csv
```

Les données xBD/xView2 complètes ne doivent pas être ajoutées au dépôt Git.

### Exemples de démo locaux

Le dossier versionné `sample_data/demo_pairs/` contient quelques paires pré/post légères pour tester le prototype sans dataset complet.

Le dossier `demo_assets/jalon5_demo_pairs/` reste local et ignoré par Git par défaut, car il contient des exports plus nombreux générés par le script de sélection.

## Vérification de taille

Commandes utiles :

```powershell
Get-ChildItem outputs\checkpoints -Recurse -File -Include *.pt,*.pth |
  Sort-Object Length -Descending |
  Select-Object FullName,@{Name='SizeMB';Expression={[math]::Round($_.Length/1MB,2)}}
```

```powershell
Get-ChildItem demo_assets -Recurse -File |
  Measure-Object -Property Length -Sum
```
