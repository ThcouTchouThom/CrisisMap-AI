# Résultats finaux et choix du modèle intégré

Ce fichier rassemble les résultats finaux disponibles à citer dans la présentation finale. Il sert aussi à expliquer pourquoi l'application conserve le modèle actuellement intégré, même si un dernier run est marginalement meilleur.

## Références damage

| Modèle | F1 damaged | IoU damaged | Mean IoU | Commentaire |
| --- | ---: | ---: | ---: | --- |
| Baseline U-Net + TTA d4 | 0.6313 | 0.4612 | 0.6816 | Baseline forte, poussée loin pour servir de référence crédible |
| Ancien champion Siamese | 0.6788 | 0.5138 | 0.7073 | Premier saut important au-delà du U-Net |
| Champion intégré dans l'app | 0.7013 | 0.5400 | 0.7283 | Modèle stabilisé, portable et testé dans l'application |
| Dernier run marginalement meilleur | 0.7018 | 0.5406 | 0.7273 | Très léger gain en F1/IoU damaged, mean IoU légèrement plus bas |

## Baseline U-Net + TTA d4

Référence :

- modèle : U-Net;
- inférence : TTA d4;
- F1 damaged : **0.6313**;
- IoU damaged : **0.4612**;
- mean IoU : **0.6816**.

Message :

> Le U-Net a été poussé assez loin pour devenir une baseline forte. Cela rend la comparaison plus crédible : les nouveaux modèles ne battent pas une baseline faible, mais une référence déjà optimisée.

## Ancien champion Siamese

Référence :

- expérience : `dlong100_hist1000_attention_safe_sqrt4_focal_tversky`;
- architecture : Siamese Attention;
- F1 damaged : **0.6788**;
- IoU damaged : **0.5138**;
- mean IoU : **0.7073**.

Message :

> Le passage aux architectures Siamese Attention a apporté un gain net, parce que ces modèles exploitent mieux la structure pré/post catastrophe.

## Champion intégré dans l'application

Référence :

- expérience : `dftv2_hist1000_attention_sqrt2_ft_250_seed0`;
- architecture : `siamese_unet_attention`;
- split : hist1000;
- loss : focal-Tversky;
- sampler : damage-sqrt alpha 2;
- epochs : 250;
- F1 damaged : **0.7013**;
- IoU damaged : **0.5400**;
- mean IoU : **0.7283**.

Ce modèle est celui intégré dans l'application de démonstration.

Raisons :

- il est performant;
- il est stabilisé;
- il dispose d'un checkpoint portable;
- il a été testé dans l'application;
- il fonctionne avec le pipeline de démonstration;
- le gain du dernier run disponible est marginal.

## Dernier run disponible marginalement meilleur

Référence :

- expérience : `dftv2_hist1000_attention_sqrt4_ft_400_seed0`;
- F1 damaged : **0.7018**;
- IoU damaged : **0.5406**;
- mean IoU : **0.7273**.

Ce run est très légèrement meilleur sur F1 damaged et IoU damaged, mais son gain est faible :

- +0.0005 en F1 damaged environ;
- +0.0006 en IoU damaged environ;
- mean IoU légèrement inférieur.

## Décision finale

Décision :

> Ne pas changer le modèle intégré dans l'application pour un gain aussi marginal.

Justification :

- le modèle intégré est déjà stable;
- il est déjà testé dans l'interface;
- il est disponible en version portable;
- le dernier run ne change pas l'histoire scientifique du projet;
- la priorité pour la présentation est la fiabilité de la démo.

Formulation recommandée :

> Le dernier run disponible est marginalement meilleur sur F1 damaged, mais le gain est trop faible pour justifier un changement de modèle juste avant la présentation. Nous gardons donc le modèle intégré, qui est stabilisé, portable et testé dans l'application.

## Champion building

Référence :

- expérience : `b400_effb4_sampler8_ft`;
- architecture : U-Net++ EfficientNet-B4;
- F1 building : **0.8504**;
- IoU building : **0.7398**.

Rôle dans le pipeline :

- prédire un masque bâtiment;
- réduire les faux positifs hors bâtiment;
- permettre le post-processing par composante;
- améliorer la lisibilité du damage final.

Message :

> Le champion building rend le pipeline plus cohérent spatialement. Il ne remplace pas le modèle damage, mais il améliore la carte finale en contraignant les prédictions autour des bâtiments.

## Message global pour le deck

Le message à faire ressortir :

> Aftermath a progressé d'une baseline U-Net solide vers un modèle Siamese Attention nettement meilleur, puis vers un pipeline complet qui combine damage, TTA, segmentation bâtiment et post-processing. Le choix final privilégie un modèle performant mais surtout fiable pour la démonstration.
