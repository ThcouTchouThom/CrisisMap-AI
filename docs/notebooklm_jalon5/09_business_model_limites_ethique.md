# Business model, limites et éthique

Ce document fournit les éléments à intégrer dans la présentation finale pour montrer qu'Aftermath / CrisisMap AI n'est pas seulement un modèle, mais aussi une proposition de produit réaliste.

## Positionnement produit

Aftermath est un outil d'aide à la décision post-catastrophe. Il analyse des images satellite pré/post événement pour produire une carte visuelle des bâtiments intacts et endommagés.

Le produit ne remplace pas une cellule de crise, un expert SIG ou une équipe terrain. Il accélère la première lecture des dégâts et aide à prioriser les zones qui méritent une inspection plus poussée.

Message produit :

> Aftermath aide les organisations à voir plus vite où agir, à partir d'images satellite déjà disponibles.

## Utilisateurs cibles

Les utilisateurs visés sont des organisations qui doivent prendre des décisions rapidement après une catastrophe :

- ONG humanitaires;
- sécurité civile;
- collectivités locales;
- gouvernements régionaux;
- assureurs;
- cellules de crise;
- équipes SIG;
- opérateurs d'infrastructures.

Ces acteurs ont souvent accès à des images satellite ou à des partenaires capables d'en fournir, mais l'analyse visuelle à grande échelle demande du temps et de l'expertise.

## Proposition de valeur

Aftermath apporte de la valeur sur quatre axes :

- **accélération** : produire une première évaluation visuelle plus vite qu'une analyse manuelle exhaustive;
- **priorisation** : aider à identifier les zones où les dégâts semblent les plus importants;
- **réduction de charge SIG** : automatiser une partie du pré-tri visuel;
- **support de communication** : fournir des cartes et overlays compréhensibles par des décideurs non spécialistes.

La promesse n'est pas d'obtenir immédiatement une vérité parfaite, mais de fournir un signal utile et explicable pour orienter l'action.

## Modèle d'affaires possible

### SaaS B2B / B2G

Modèle principal envisageable :

- abonnement annuel pour organisations;
- accès à une interface web;
- nombre limité ou illimité d'analyses selon le contrat;
- support et maintenance inclus.

Ce modèle convient aux assureurs, collectivités, agences publiques et opérateurs d'infrastructures.

### Licence annuelle institutionnelle

Pour les acteurs publics ou parapublics :

- licence annuelle par organisation;
- déploiement contrôlé;
- conformité et sécurité renforcées;
- possibilité d'hébergement privé.

### Facturation par crise

Pour les ONG ou cellules de crise :

- usage ponctuel lors d'un événement;
- facturation par zone analysée ou par lot d'images;
- modèle plus flexible pour les structures qui n'ont pas besoin d'un abonnement permanent.

### Intégration SIG / API

À moyen terme, Aftermath peut être proposé comme composant intégré :

- API de prédiction;
- export GeoJSON ou raster;
- intégration QGIS / ArcGIS;
- connexion à des workflows de cartographie existants.

Cette option est importante car les équipes terrain utilisent souvent déjà des outils SIG.

## Différenciation

Aftermath se différencie par :

- l'utilisation directe de paires pré/post catastrophe;
- une sortie visuelle lisible;
- une combinaison damage + segmentation bâtiment;
- une interface démontrable;
- une logique d'explicabilité visuelle;
- une orientation produit, pas seulement recherche.

## Limites techniques

### Qualité des images satellite

Le modèle dépend fortement de la qualité des images :

- résolution;
- angle de vue;
- luminosité;
- couverture nuageuse;
- délai d'acquisition;
- alignement entre image pré et image post.

Si l'image post-catastrophe est mauvaise ou trop différente de l'image pré-catastrophe, la prédiction peut être moins fiable.

### Généralisation

Le dataset xBD / xView2 couvre plusieurs catastrophes et zones géographiques, mais la généralisation reste un enjeu :

- nouveaux pays;
- nouveaux types de bâtiments;
- catastrophes peu représentées;
- milieux ruraux ou urbains très denses;
- matériaux et architectures locales.

Un produit réel devrait être évalué sur des zones et événements non vus.

### Erreurs du modèle building

La segmentation bâtiment améliore le pipeline, mais elle peut aussi introduire des erreurs :

- un bâtiment raté peut supprimer un vrai dommage;
- un faux bâtiment peut créer une fausse zone d'analyse;
- les contours peuvent être imparfaits.

C'est pourquoi le masque bâtiment doit être présenté comme une aide et non comme une vérité absolue.

### Limitation actuelle à 3 classes

Le pipeline actuel utilise trois classes :

- fond;
- bâtiment intact;
- bâtiment endommagé.

Ce n'est pas encore le score officiel xView2 complet à 5 classes :

- no damage;
- minor damage;
- major damage;
- destroyed.

La transition vers 5 classes est une étape future importante, mais le travail actuel permet déjà de valider la chaîne globale.

### Besoin de supervision humaine

Aftermath doit rester un outil d'assistance :

- un expert doit pouvoir vérifier les résultats;
- l'incertitude doit être visible;
- les décisions critiques ne doivent pas dépendre uniquement du modèle.

## Limites éthiques

### Ne pas remplacer la décision humaine

Dans un contexte humanitaire ou de sécurité civile, une erreur de priorisation peut avoir des conséquences importantes. Le système doit donc être présenté comme une aide à l'analyse, pas comme un décideur automatique.

Phrase à utiliser :

> Aftermath propose une carte d'aide à la décision, pas une décision automatique.

### Risque de mauvaise priorisation

Un modèle peut :

- manquer une zone réellement endommagée;
- surestimer des dégâts;
- orienter l'attention vers une zone moins prioritaire;
- donner une impression de certitude excessive.

La présentation doit reconnaître ce risque clairement.

### Transparence sur l'incertitude

L'interface doit aider l'utilisateur à comprendre :

- où le modèle est confiant;
- où il hésite;
- quels masques intermédiaires influencent la décision finale;
- quelles images ont été utilisées.

L'onglet d'incertitude et les visualisations intermédiaires sont donc importants.

### Usage responsable

Dans un contexte humanitaire, il faut éviter :

- les décisions opaques;
- la surconfiance dans une carte IA;
- l'utilisation hors contexte;
- l'absence de validation terrain.

L'usage responsable implique une boucle humaine et une documentation claire des limites.

## Message final à intégrer

Aftermath a un potentiel produit réel, mais son intérêt vient justement de son positionnement responsable :

> accélérer l'analyse, prioriser l'attention, rendre les prédictions visibles, et laisser la décision finale aux humains.
