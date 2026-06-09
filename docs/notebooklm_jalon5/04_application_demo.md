# Application de démonstration

## Objectif de l'application

L'application Streamlit sert à montrer le pipeline Aftermath de manière interactive.

Elle doit permettre à un utilisateur de :

- choisir un modèle ;
- choisir un mode de pipeline ;
- utiliser un exemple xBD ou téléverser une paire réelle ;
- lancer l'inférence ;
- visualiser la carte de dégâts ;
- comprendre le rôle du masque bâtiment ;
- exporter les résultats.

## Modes disponibles

L'application propose deux sources de données :

1. **Téléverser des images** ;
2. **Exemples du dataset**.

Le mode upload est important pour la présentation, car il montre que l'application ne dépend pas uniquement d'un exemple codé en dur.

## Pipelines de démonstration

Trois pipelines sont proposés :

| Mode | Description |
| --- | --- |
| Rapide : damage seul | Inférence damage brute, rapide |
| Qualité : damage + TTA d4 | Inférence plus stable avec TTA |
| Qualité maximale : damage + TTA d4 + building post-process | Pipeline recommandé si CUDA et checkpoint building sont disponibles |

Le mode qualité maximale est celui à privilégier pour la démonstration classe.

## Modèle damage intégré

Le modèle principal actuel est :

`dftv2_hist1000_attention_sqrt2_ft_250_seed0`

Résumé :

- architecture : `siamese_unet_attention` ;
- F1 damaged : 0.701317 ;
- IoU damaged : 0.540022 ;
- mean IoU : 0.728266.

L'application doit aussi garder les anciens U-Net comme fallback.

## Modèle building intégré

Le modèle building actuel est :

`b400_effb4_sampler8_ft`

Résumé :

- architecture : U-Net++ EfficientNet-B4 ;
- F1 building : 0.850421 ;
- IoU building : 0.739767.

## Visualisations attendues

Dans l'onglet visualisation, l'application doit montrer :

- image avant catastrophe ;
- image après catastrophe ;
- damage brut ou damage avec TTA ;
- masque bâtiment prédit ;
- damage final post-processé ;
- vérité terrain si elle est disponible ;
- overlay final sur l'image post-catastrophe.

Ces éléments rendent le pipeline explicable visuellement.

## Métriques dans l'application

En mode dataset, la vérité terrain est disponible. L'application peut donc afficher :

- pixel accuracy ;
- mean IoU ;
- IoU damaged ;
- F1 damaged ;
- distribution des classes.

En mode upload, il n'y a généralement pas de vérité terrain. L'application affiche donc des statistiques de prédiction, mais pas de métriques de performance.

## Onglet incertitude

L'application inclut une visualisation d'incertitude par entropie. Cela aide à expliquer que le modèle peut être très confiant dans certaines zones et plus hésitant dans d'autres.

Cette carte est utile pour parler d'explicabilité :

- contours de bâtiments ;
- textures ambiguës ;
- zones où intact et endommagé sont difficiles à distinguer.

## Déroulé conseillé pour la démo

1. Ouvrir l'application.
2. Montrer la sidebar :
   - modèle damage champion ;
   - modèle building b400 ;
   - pipeline recommandé.
3. Utiliser le mode dataset ou upload.
4. Lancer l'inférence.
5. Montrer le slider post-image / overlay final.
6. Montrer les panneaux :
   - damage brut ;
   - masque bâtiment ;
   - damage final.
7. Expliquer que le masque bâtiment rend la décision plus cohérente.
8. Montrer les métriques si un exemple dataset est utilisé.

## Message oral

L'application n'est pas seulement une interface de visualisation. Elle est la preuve que le pipeline complet peut prendre une paire pré/post, faire tourner le modèle et produire une carte compréhensible par un utilisateur non spécialiste.
