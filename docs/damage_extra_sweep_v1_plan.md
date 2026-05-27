# Plan - campagne damage extra 1024

## Contexte

La campagne `long250` damage s'est terminee proprement avec 5 jobs de 250 epochs. Le nouveau meilleur resultat est :

- experience : `unet_1024_long250_noleak_match_hist_all_aug-safe_sampler-damage-sqrt-alpha4_250epochs`
- split : `splits_noleak_match_hist_all`
- augmentation : `safe`
- sampler : `damage-sqrt`, alpha `4`
- mean IoU : `0.676624`
- IoU damaged : `0.446452`
- precision damaged : `0.605233`
- recall damaged : `0.629871`
- F1 damaged : `0.617307`

Ce modele depasse le champion precedent reproduit sur `match_hist1000 + none + none + 250 epochs` :

- mean IoU : `0.666015`
- IoU damaged : `0.424620`
- F1 damaged : `0.596116`

## Objectif

Cette campagne extra teste cinq configurations ciblees, sans relancer une grille exhaustive. Le but est de verifier si la recette gagnante se generalise a d'autres splits, si une variante `damage-aware` peut encore progresser, et si une duree plus longue ameliore le nouveau champion.

## Configurations

1. `match_hist_all + safe + damage-sqrt alpha4`, pousse a `500 epochs`.
   - C'est la recette championne.
   - Le run utilise un nouveau nom `extra500` afin de ne pas ecraser le checkpoint 250 epochs termine.
   - Le run peut repartir de zero; reprendre depuis le 250 epochs n'est pas requis.

2. `match_hist_all + damage-aware + damage-sqrt alpha4`, `250 epochs`.
   - Teste si l'augmentation orientee dommages fonctionne mieux quand elle est combinee au sampler gagnant.

3. `dmg001_v2 + safe + damage-sqrt alpha4`, `250 epochs`.
   - Teste la recette gagnante sur un split plus oriente degats.
   - Ce split peut favoriser le rappel, avec un risque de baisse de precision.

4. `match_hist1000 + safe + damage-sqrt alpha4`, `250 epochs`.
   - Teste la recette gagnante sur le split du champion historique.
   - Permet une comparaison propre avec l'ancien controle `none + none`.

5. `match_hist_all + safe + damage-sqrt alpha8`, `250 epochs`.
   - Test alpha8 limite a une seule configuration, car augmenter alpha peut vite favoriser trop fortement le rappel au detriment de la precision.
   - Si alpha8 ameliore IoU/F1, une campagne plus fine pourra tester alpha6 ou alpha10.

## Parametres communs

- modele : U-Net local
- resolution : `1024`
- batch size : `2`
- loss : `ce-dice`
- class weights : `0.05 1.0 4.0`
- learning rate : `1e-4`
- target mode : `3-class`
- AMP : active automatiquement sur CUDA par `train_unet.py`
- workers Rorqual : `4`

## Scripts

- `configs/damage_extra_sweep_v1.csv` : configuration des 5 runs.
- `slurm/run_damage_extra_config.sh` : runner generique une ligne CSV / un job.
- `slurm/submit_damage_extra_sweep_v1.sh` : soumet les 5 jobs independamment.
- `scripts/rebuild_damage_extra_summary.py` : reconstruit le resume global.

Le resume est ecrit ici :

```text
outputs/predictions/unet_1024_damage_extra_sweep_v1_summary.csv
```

## Strategie de soumission

Par defaut, les 5 jobs sont independants :

```bash
bash slurm/submit_damage_extra_sweep_v1.sh
```

Une dependance optionnelle peut etre ajoutee si on veut attendre la campagne Building100 :

```bash
WAIT_FOR_BUILDING100=1 BUILDING100_DEPENDENCIES=<jobid[:jobid...]> bash slurm/submit_damage_extra_sweep_v1.sh
```

## Relaunch apres timeout

Les premiers jobs `damage_extra` ont atteint la limite de temps, mais les checkpoints partiels sont exploitables. Chaque dossier contient `last_unet.pt` avec :

- `epoch`
- `model_state_dict`
- `optimizer_state_dict`
- `metrics`
- `config`
- `loss_config`

La reprise est donc possible avec `--resume-checkpoint`, sans evaluer les checkpoints partiels comme resultats officiels.

Commande recommandee :

```bash
bash slurm/submit_damage_extra_resume.sh
```

Cette commande utilise `configs/damage_extra_sweep_v1_resume.csv`, avec `08:00:00` pour le run `500 epochs` et `04:00:00` pour les autres runs. Elle soumet les memes 5 configurations avec `RESUME_INCOMPLETE=1`.

Regles de securite :

- run complet + metriques test presentes : skip ;
- historique complet mais metriques test absentes : evaluation seulement ;
- run incomplet + `RESUME_INCOMPLETE=1` + `last_unet.pt` : reprise ;
- run incomplet + `FORCE_INCOMPLETE=1` : suppression du dossier incomplet puis reentrainement ;
- run incomplet sans option explicite : arret propre, pas d'evaluation.

Les metriques officielles ne sont produites qu'apres entrainement complet et evaluation test.

Audit :

```bash
python scripts/audit_campaign_completion.py --campaign damage_extra
```

## Comparaison attendue

Les resultats seront compares a deux references :

1. nouveau champion long250 :
   - `match_hist_all + safe + damage-sqrt alpha4 + 250 epochs`
   - IoU damaged `0.446452`
   - F1 damaged `0.617307`

2. champion historique reproduit :
   - `match_hist1000 + none + none + 250 epochs`
   - IoU damaged `0.424620`
   - F1 damaged `0.596116`

Priorite d'analyse :

1. IoU damaged
2. F1 damaged
3. rappel damaged
4. precision damaged
5. mean IoU

## Lien avec Building100

La campagne Building100 segmentation batiment tourne separement. Elle sera analysee plus tard pour decider si un pipeline en deux etapes peut depasser durablement le modele damage brut. Cette campagne extra reste centree sur le modele damage U-Net actuel et ne modifie pas l'architecture.
