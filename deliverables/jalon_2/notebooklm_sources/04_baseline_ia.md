# Baseline IA

## Modèle

Le baseline est un U-Net léger pour segmentation sémantique.

Entrée :

- image RGB pré-catastrophe ;
- image RGB post-catastrophe ;
- concaténation en tenseur 6 canaux.

Sortie :

- carte de segmentation 3 classes : background, no damage, damaged.

Le modèle utilisé dans le MVP compte environ 7,763,971 paramètres pour la configuration de référence.

## Fonction de perte

La Cross-Entropy seule fonctionne, mais elle est sensible au déséquilibre de classes. La variante de référence combine :

- Cross-Entropy pondérée ;
- Dice loss multiclasses.

Les poids de référence pour la formulation 3 classes sont :

```text
[0.05, 1.0, 4.0]
```

Interprétation :

- background fortement réduit ;
- bâtiment non endommagé gardé comme classe intermédiaire ;
- damaged renforcé pour compenser sa rareté.

## Expériences réalisées

Des expériences locales ont testé :

- pertes : CE, weighted CE, CE-Dice ;
- poids de classes ;
- learning rate ;
- image size 512 ;
- premiers essais 1024 ;
- batch sizes 1, 2, 3 et 4 selon la mémoire disponible.

La combinaison CE-Dice + `[0.05, 1.0, 4.0]` est devenue la référence, car elle donne un meilleur compromis sur la classe damaged.

## État actuel

Pour le jalon 2, le baseline U-Net suffit à démontrer :

- que les données sont chargées ;
- que le modèle apprend ;
- que les prédictions sont visualisables ;
- que les métriques sont calculées.

Les architectures plus avancées sont réservées aux étapes suivantes.

