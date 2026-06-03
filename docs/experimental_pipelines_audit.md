# Audit des nouveaux pipelines expérimentaux

## Portée

Cet audit couvre les nouvelles briques expérimentales créées avant lancement Rorqual :

- export et métriques xView2-style;
- campagne `xView2 strong baseline`;
- campagne `Multi-Temporal Fusion`.

Il ne modifie pas les données, checkpoints, sorties existantes, logs ou résultats déjà complétés.

## Fichiers audités

Scripts Python :

- `scripts/export_xview2_format.py`
- `scripts/evaluate_xview2_style_metrics.py`
- `src/crisismap/models/multitemporal_fusion.py`
- `scripts/train_multitemporal_fusion.py`
- `scripts/evaluate_multitemporal_fusion.py`
- `src/crisismap/models/xview2_strong_baseline.py`
- `scripts/train_xview2_strong_baseline.py`
- `scripts/evaluate_xview2_strong_baseline.py`

Configurations :

- `configs/multitemporal_fusion_sweep_v1.csv`
- `configs/xview2_strong_baseline_sweep_v1.csv`

Scripts SLURM :

- `slurm/run_multitemporal_fusion_config.sh`
- `slurm/submit_multitemporal_fusion_sweep_v1.sh`
- `slurm/run_xview2_strong_baseline_config.sh`
- `slurm/submit_xview2_strong_baseline_sweep_v1.sh`
- `slurm/smoke_multitemporal_fusion.sbatch`
- `slurm/smoke_xview2_strong_baseline.sbatch`

## Script d'audit

Le script suivant effectue une vérification statique :

`scripts/audit_new_experimental_pipelines.py`

Il vérifie notamment :

- existence des fichiers attendus;
- absence de chemins Windows locaux codés en dur;
- CSV simples, séparés par virgules, sans champs quotés;
- noms d'expériences uniques, donc dossiers checkpoints distincts;
- usage de `${SCRATCH}/CrisisMap-AI` pour logs/cache SLURM;
- absence de partition explicite;
- présence de `mkdir -p`;
- notifications email SLURM;
- logique runner : skip run complet, evaluate-only, refus des runs partiels, `FORCE_INCOMPLETE=1`;
- présence AMP dans les scripts d'entraînement;
- mapping 3 classes documenté;
- mode futur 5 classes documenté sans prétendre à un score officiel en mode 3 classes;
- cohérence des métriques avec `metrics_from_confusion`.

Commande :

```bash
python scripts/audit_new_experimental_pipelines.py
```

Optionnellement, si les dépendances sont installées, le script peut instancier plusieurs modèles sur CPU :

```bash
python scripts/audit_new_experimental_pipelines.py --run-model-smoke
```

## Smoke tests Rorqual

Deux jobs SLURM légers ont été ajoutés.

### xView2 strong baseline

```bash
sbatch slurm/smoke_xview2_strong_baseline.sbatch
```

Ce test :

- demande un H100;
- charge les modules Rorqual habituels;
- active `~/virtualenvs/crisismap-ai`;
- instancie les modèles ResNet34/ResNet50 strong-baseline;
- exécute un forward dummy `[1, 6, 256, 256]`;
- vérifie les shapes :
  - `building_logits = [1, 1, 256, 256]`;
  - `damage_logits = [1, C, 256, 256]`.

### Multi-Temporal Fusion

```bash
sbatch slurm/smoke_multitemporal_fusion.sbatch
```

Ce test :

- instancie les variantes MTF FPN, DeepLab-style, EfficientNet-B3;
- instancie les contrôles 6 canaux et Siamese attention;
- exécute un forward dummy `[1, 6, 256, 256]`;
- vérifie les shapes de sortie bâtiment et damage.

Ces smoke tests ne lisent pas le dataset et ne créent pas de checkpoints.

## Points corrigés pendant l'audit

Le runner `slurm/run_xview2_strong_baseline_config.sh` a été renforcé :

- si l'historique contient le nombre attendu d'epochs et que les métriques test manquent, le script évalue seulement le checkpoint complet;
- si l'historique est complet mais que `best_xview2_strong.pt` manque, le script refuse le run au lieu de réentraîner silencieusement;
- `FORCE_INCOMPLETE=1` est supporté pour réentraîner volontairement un run incomplet.

## Mapping des cibles

Le mode actuel 3 classes est :

```text
0 = background
1 = no_damage
2 = damaged
```

Le mode futur 5 classes documenté est :

```text
0 = background
1 = no_damage
2 = minor
3 = major
4 = destroyed
```

Les scripts xView2-style nomment explicitement le score 3 classes :

`binary_damage_xview2_like_score`

Ce score n'est pas comparable au leaderboard xView2 officiel.

## Gestion des sorties

Les nouvelles campagnes écrivent dans des dossiers distincts :

- `outputs/checkpoints/<experiment>/`
- `outputs/predictions/xview2_strong_baseline/`
- `outputs/predictions/multitemporal_fusion/`

Les CSV utilisent des noms d'expérience uniques pour éviter tout chevauchement entre runs.

## Logique de sécurité des runners

Les runners sont conçus pour :

- sauter les runs complets;
- évaluer un checkpoint complet si les métriques test manquent;
- refuser les checkpoints partiels par défaut;
- reprendre les runs MTF partiels avec `RESUME_INCOMPLETE=1`;
- réentraîner volontairement un run incomplet avec `FORCE_INCOMPLETE=1`.

Les runs `xView2 strong baseline` ne reprennent pas encore depuis checkpoint partiel; ils refusent les runs partiels par défaut et demandent `FORCE_INCOMPLETE=1` pour repartir de zéro.

## Limites à garder en tête

- Les modèles DeepLab-style peuvent être plus sensibles aux petits batchs avec BatchNorm; la campagne MTF utilise des crops 512 et batch 2, ce qui devrait éviter le cas `[1, C, 1, 1]` observé ailleurs, mais le smoke test reste utile.
- Les exports xView2-style sont mask-based et internes au projet.
- Les scripts ne lancent aucune soumission officielle xView2.
- Les campagnes v1 sont exploratoires; elles ne remplacent pas le champion U-Net tant qu'elles ne dépassent pas ses métriques no-leak.
