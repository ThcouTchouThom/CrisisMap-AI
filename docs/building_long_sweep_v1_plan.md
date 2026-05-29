# Plan Building Long Sweep v1

## Objectif

Building100 est termine avec 49/49 runs. Cette campagne longue ne relance pas
Building100 : elle entraine seulement cinq candidats choisis pour verifier si
les meilleurs signaux a 100 epochs tiennent sur 250 ou 400 epochs.

Les sorties sont volontairement distinctes :

```text
outputs/checkpoints/<experiment>/
outputs/predictions/<experiment>_building_test_metrics.json
outputs/predictions/building_long_sweep_v1_summary.csv
```

## Candidats

| Experiment | Pourquoi |
| --- | --- |
| `b250_effb4_sampler8_ft` | Meilleur modele equilibre Building100. |
| `b400_effb4_sampler8_ft` | Meme recette poussee plus longtemps pour tester la stabilite. |
| `b250_effb3_building_rich_ft` | Runner-up tres proche, moins lourd que EfficientNet-B4. |
| `b250_effb3_dmg001_ft` | Split oriente damage avec tres bon rappel batiment. |
| `b250_deeplab_resnet50_sampler8_ft` | Modele haut rappel, utile pour ne pas supprimer de vrais batiments en aval. |

## Resultats Building100 utilises pour choisir

| Source Building100 | Building IoU | F1 | Recall |
| --- | ---: | ---: | ---: |
| `b100_d_full_pre_unetplusplus_effb4_sampler8_focal_tversky` | 0.703165 | 0.825716 | 0.872600 |
| `b100_f_building_rich_002_pre_unetplusplus_effb3_focal_tversky` | 0.700954 | 0.824189 | 0.870745 |
| `b100_d_full_pre_deeplabv3plus_resnet50_sampler8_focal_tversky` | 0.684301 | 0.812564 | 0.906086 |
| `b100_f_dmg001_v2_pre_unetplusplus_effb3_focal_tversky` | 0.699264 | 0.823020 | 0.891133 |

## Time limits

Les `metrics_history.json` Building100 n'etaient pas disponibles localement au
moment de preparer cette campagne, donc les durees ne peuvent pas etre calculees
depuis les temps reels par epoch sur cette machine. Les limites du CSV sont
derivees des budgets Building100 correspondants, multiplies par le nombre
d'epochs demande, avec une marge de securite.

| Experiment | Base Building100 | Estimation |
| --- | --- | --- |
| `b250_effb4_sampler8_ft` | 6h30 pour 100 epochs | 19h pour 250 epochs |
| `b400_effb4_sampler8_ft` | 6h30 pour 100 epochs | 30h pour 400 epochs |
| `b250_effb3_building_rich_ft` | 4h30 pour 100 epochs | 14h pour 250 epochs |
| `b250_effb3_dmg001_ft` | 4h30 pour 100 epochs | 14h pour 250 epochs |
| `b250_deeplab_resnet50_sampler8_ft` | 5h pour 100 epochs | 16h pour 250 epochs |

Si la partition refuse une limite longue, reduire `time_limit` dans le CSV et
relancer avec :

```bash
RESUME_INCOMPLETE=1 bash slurm/submit_building_long_sweep_v1.sh
```

## Securite

- Un run est complet seulement si `metrics_history.json` contient le nombre
  d'epochs attendu et si les metriques test JSON/CSV existent.
- Un checkpoint partiel n'est jamais evalue comme resultat officiel.
- `FORCE_INCOMPLETE=1` supprime seulement un run incomplet avant de le refaire.
- `RESUME_INCOMPLETE=1` reprend depuis `last_building.pt` si disponible.
- DeepLabV3+ utilise `--drop-last-train` pour eviter le BatchNorm sur le dernier
  batch de taille 1.

## Commandes

Soumettre les cinq jobs independants :

```bash
bash slurm/submit_building_long_sweep_v1.sh
```

Reconstruire le resume :

```bash
python scripts/rebuild_building_long_summary.py
```

Le resume principal est :

```text
outputs/predictions/building_long_sweep_v1_summary.csv
```
