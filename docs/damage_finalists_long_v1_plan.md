# Plan Damage Finalists Long v1

## Objectif

La campagne `damage_arch_v2` est terminee. Elle montre que les architectures
Siamese change-aware depassent deja le precedent U-Net a 100 epochs. Cette
campagne cible six finalistes sans modifier les sorties v2 existantes.

Les nouveaux resultats sont separes :

```text
outputs/checkpoints/<experiment>/
outputs/predictions/<experiment>_test_metrics.json
outputs/predictions/damage_finalists_long_v1_summary.csv
```

## Reference

Ancien champion U-Net avec TTA d4 :

| Metrique | Valeur |
| --- | ---: |
| Mean IoU | 0.681574 |
| IoU damaged | 0.461240 |
| F1 damaged | 0.631300 |

Meilleurs runs `damage_arch_v2` a 100 epochs :

| Run | F1 damaged | IoU damaged | Raison de selection |
| --- | ---: | ---: | --- |
| `damage_arch_v2_hist1000_attention_safe_sqrt4_ce_dice` | 0.641743 | 0.472476 | Meilleur F1 et meilleur IoU damaged. |
| `damage_arch_v2_histall_attention_safe_sqrt4_ce_dice` | 0.639457 | 0.470002 | Meilleure mean IoU parmi les runs v2. |
| `damage_arch_v2_hist1000_abs_signed_safe_sqrt4_focal_tversky` | 0.636222 | 0.466514 | Bon compromis precision/rappel. |

## Runs

| Experiment | But |
| --- | --- |
| `dlong250_hist1000_attention_safe_sqrt4_ce_dice` | Confirmer le meilleur run v2 sur 250 epochs. |
| `dlong400_hist1000_attention_safe_sqrt4_ce_dice` | Tester si le meilleur run continue a progresser sur 400 epochs. |
| `dlong250_histall_attention_safe_sqrt4_ce_dice` | Comparer le split `match_hist_all` plus diversifie. |
| `dlong250_hist1000_abs_signed_safe_sqrt4_focal_tversky` | Conserver une architecture de fusion explicite alternative. |
| `dlong100_hist1000_attention_safe_sqrt4_focal_tversky` | Tester focal-Tversky sur la meilleure architecture. |
| `dlong100_hist1000_attention_safe_sqrt4_focal_dice` | Tester focal-Dice sur la meilleure architecture. |

## Time limits

Les historiques v2 n'etaient pas disponibles dans le workspace local lors de
la preparation. Les limites sont donc derivees des budgets v2 de 5h a 5h30 pour
100 epochs, multiplies par le nombre d'epochs, avec marge.

| Type de run | Limite |
| --- | --- |
| Attention 250 epochs | `15:00:00` |
| Attention 400 epochs | `24:00:00` |
| Abs-signed focal-Tversky 250 epochs | `14:00:00` |
| Variantes loss exploratoires 100 epochs | `06:00:00` |

Si une limite est insuffisante, relancer uniquement les runs incomplets avec :

```bash
RESUME_INCOMPLETE=1 bash slurm/submit_damage_finalists_long_v1.sh
```

## Securite

- Les jobs sont independants.
- Les noms `dlong*` garantissent des dossiers distincts des checkpoints v2.
- Un run est complet seulement si `metrics_history.json` contient le nombre
  d'epochs attendu et si les metriques test JSON/CSV existent.
- Un checkpoint partiel n'est jamais evalue comme resultat officiel.
- `FORCE_INCOMPLETE=1` supprime uniquement les artefacts incomplets avant un
  nouvel entrainement.
- Les logs runtime et le cache Triton utilisent `${SCRATCH}/CrisisMap-AI`.
- Le runner ne fixe pas explicitement de partition SLURM.

## Commandes

Soumettre les six jobs independants :

```bash
bash slurm/submit_damage_finalists_long_v1.sh
```

Reconstruire le resume :

```bash
python scripts/rebuild_damage_finalists_long_summary.py
```
