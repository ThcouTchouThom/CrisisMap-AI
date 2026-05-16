# Résultats synthèse jalon 2

## Baseline minimal

| Expérience | Résolution | Split | Epochs | Pixel accuracy | Mean IoU | IoU damaged | F1 damaged |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| U-Net baseline old4 | 512 | `data/processed/splits` | 30 | 0.9175 | 0.6257 | 0.3870 | 0.5581 |

Ce résultat est suffisant pour démontrer un baseline IA fonctionnel au jalon 2.

## Résultats avancés propres no-leak

| Expérience | Résolution | Protocole | Mean IoU | IoU damaged | F1 damaged |
| --- | ---: | --- | ---: | ---: | ---: |
| U-Net CE-Dice, `match_hist1000` | 1024 | no-leak, 100 epochs | ~0.6578 | ~0.4159 | ~0.5874 |
| U-Net CE-Dice, `match_hist1000` | 1024 | no-leak, 250 epochs | ~0.6651 | ~0.4175 | ~0.5891 |

Ces résultats sont à présenter comme preuve d'avancement, pas comme conclusion finale.

## Fichiers sources de résultats

Résultats et métriques générés localement :

- `outputs/predictions/unet_512_v2_30epochs_test_metrics.json`
- `outputs/predictions/unet_1024_noleak_splits_100epochs_summary.csv`
- `outputs/predictions/unet_long_best_models_noleak_summary.csv`

Les fichiers `outputs/` ne sont pas destinés à être suivis par Git.

