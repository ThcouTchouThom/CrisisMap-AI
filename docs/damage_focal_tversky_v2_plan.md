# Campagne damage_focal_tversky_v2

## Objectif

Cette campagne explore largement mais de manière structurée la famille damage actuellement gagnante :

```text
dlong100_hist1000_attention_safe_sqrt4_focal_tversky
```

Référence actuelle :

| Modèle | F1 damaged | IoU damaged | Mean IoU |
| --- | ---: | ---: | ---: |
| U-Net + TTA d4 | 0.631300 | 0.461240 | 0.681574 |
| Champion attention focal-tversky actuel | 0.678801 | 0.513776 | 0.707285 |

Le résultat `abs_signed` long a moins bien fonctionné que l'attention. La priorité est donc de comprendre si l'attention focal-tversky continue à progresser et si elle est robuste.

## Design expérimental

La campagne contient 33 runs maximum. Elle reste centrée sur les variantes réellement pertinentes, sans relancer une recherche aléatoire.

### A. Attention focal-tversky hist1000 sqrt4

Ces runs testent la progression avec plus d'epochs et la stabilité par seed :

- 100, 250, 400, 600 epochs seed0 ;
- 100 epochs seed1 / seed2 ;
- 250 epochs seed1 / seed2.

Le run 600 epochs est inclus pour vérifier si la courbe continue à progresser, avec un `time_limit` élevé mais borné.

### B. Sampler alpha

Le déséquilibre de la classe damage reste critique. On teste :

- `damage-sqrt` alpha 2 ;
- `damage-sqrt` alpha 4 ;
- `damage-sqrt` alpha 8 ;
- aucun sampler.

Chaque variante principale est testée à 100 et 250 epochs.

### C. Split hist1000 vs histall

`splits_noleak_match_hist1000` semble très fort, mais `splits_noleak_match_hist_all` peut généraliser différemment. Les variantes histall testent alpha 2/4/8 à 100 et 250 epochs.

### D. Architectures proches

On garde des architectures proches de la famille gagnante :

- `siamese_unet_attention_base48` ;
- `siamese_unet_attention_base64` ;
- `siamese_unet_gated_fusion` ;
- `siamese_unet_abs_signed_product` ;
- `siamese_unet_shared_encoder`.

Chaque variante est testée en 100 et 250 epochs sur hist1000, safe, sqrt4, focal-tversky.

### E. Pertes de référence

On conserve quelques pertes de référence :

- focal-dice 250 et 400 ;
- ce-dice 250.

Les anciens runs ce-dice longs déjà disponibles doivent être utilisés comme comparaison historique s'ils existent, sans écraser leurs outputs.

## Fichiers

- Config : `configs/damage_focal_tversky_v2.csv`
- Runner : `slurm/run_damage_focal_tversky_v2_config.sh`
- Submitter : `slurm/submit_damage_focal_tversky_v2.sh`
- Smoke test : `slurm/smoke_damage_focal_tversky_v2.sbatch`
- Résumé : `scripts/rebuild_damage_focal_tversky_v2_summary.py`

## Outputs attendus

Checkpoints :

```text
outputs/checkpoints/<experiment>/
```

Métriques :

```text
outputs/predictions/damage_focal_tversky_v2/<experiment>_test_metrics.json
outputs/predictions/damage_focal_tversky_v2/<experiment>_test_metrics.csv
```

Résumé :

```text
outputs/predictions/damage_focal_tversky_v2_summary.csv
```

## Sécurité

Le runner :

- saute les runs complets ;
- évalue seulement si l'historique est complet mais les métriques manquent ;
- refuse les checkpoints partiels par défaut ;
- supporte `RESUME_INCOMPLETE=1` depuis `last_damage_arch.pt` ;
- supporte `FORCE_INCOMPLETE=1` pour supprimer/repartir d'un run incomplet ;
- n'écrit pas dans les dossiers de résultats existants hors préfixe `damage_focal_tversky_v2`.

## Lancement

Ne pas lancer avant d'avoir résumé les campagnes déjà terminées. Quand le lancement est décidé :

```bash
bash slurm/submit_damage_focal_tversky_v2.sh
```

Pour relancer un run partiel :

```bash
RESUME_INCOMPLETE=1 bash slurm/submit_damage_focal_tversky_v2.sh
```
