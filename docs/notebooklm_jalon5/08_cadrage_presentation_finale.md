# Cadrage de la présentation finale

Ce document sert à recadrer la présentation finale d'Aftermath / CrisisMap AI. Le premier deck généré était visuellement bon, mais la version finale doit mieux respecter les attentes pédagogiques : une présentation d'environ **15 minutes + questions**, structurée comme un **pitch de produit technologique**, avec une **démonstration centrale** et une preuve technique crédible.

## Attentes principales

La présentation finale doit montrer que le projet n'est plus seulement une expérience IA, mais une proposition de produit démontrable :

- un problème réel et important est identifié;
- un utilisateur cible est clairement défini;
- une solution concrète est proposée;
- le prototype fonctionne sur de vraies images pré/post catastrophe;
- les résultats sont mesurés avec des métriques compréhensibles;
- les limites sont reconnues;
- une trajectoire produit et technique est proposée.

La présentation doit donc équilibrer deux dimensions :

- **pitch produit** : problème, valeur, utilisateur, usage, modèle d'affaires;
- **preuve technique** : dataset, pipeline IA, modèles, métriques, explicabilité, limites.

## Principe central

La démonstration doit occuper le centre de la présentation. Elle ne doit pas être un détail en fin d'exposé, mais le moment où l'auditoire comprend concrètement ce que fait Aftermath.

Message à faire ressortir :

> Aftermath transforme une paire d'images satellite pré/post catastrophe en une carte visuelle exploitable des bâtiments intacts et endommagés.

## Structure recommandée : 12 à 14 slides maximum

### Slide 1 - Titre et promesse produit

**Aftermath - Voir les dégâts pour agir plus vite**

Durée cible : 30 secondes.

Objectif : installer immédiatement le projet, le contexte catastrophe et la promesse.

### Slide 2 - Problème terrain

Durée cible : 1 minute.

Points à couvrir :

- après une catastrophe, les premières heures sont critiques;
- les images satellite donnent une vue large, mais leur analyse reste coûteuse;
- les équipes terrain doivent prioriser rapidement;
- l'analyse manuelle complète est lente.

### Slide 3 - Utilisateurs cibles et besoin

Durée cible : 1 minute.

Utilisateurs :

- ONG;
- sécurité civile;
- collectivités;
- assureurs;
- cellules de crise;
- équipes SIG.

Besoin : obtenir rapidement une première lecture visuelle des zones touchées.

### Slide 4 - Produit proposé

Durée cible : 1 minute.

Aftermath est présenté comme une interface d'aide à la décision :

- entrée : image satellite avant catastrophe + image après catastrophe;
- sortie : carte de bâtiments intacts et endommagés;
- interface : prototype Streamlit;
- valeur : accélérer l'évaluation initiale, pas remplacer l'expertise humaine.

### Slide 5 - Données et formulation IA

Durée cible : 1 minute.

Points :

- dataset xBD / xView2;
- paires pré/post catastrophe;
- segmentation sémantique 3 classes :
  - fond;
  - bâtiment intact;
  - bâtiment endommagé;
- protocole no-leak pour éviter une évaluation artificiellement optimiste.

### Slide 6 - Pipeline technique simplifié

Durée cible : 1 minute.

Pipeline :

1. images pré/post;
2. modèle damage Siamese Attention;
3. TTA d4;
4. segmentation bâtiment;
5. post-processing par component majority;
6. overlay final dans l'application.

Message : le pipeline combine prédiction IA, stabilisation d'inférence et contrainte bâtiment pour produire une carte plus lisible.

### Slides 7 à 9 - Démonstration centrale

Durée cible : 4 à 5 minutes.

La démo doit être la partie centrale.

Déroulé :

1. ouvrir l'application classique;
2. choisir le pipeline qualité maximale;
3. charger une paire satellite;
4. lancer l'inférence;
5. lire l'overlay final;
6. montrer les sorties intermédiaires;
7. expliquer le masque bâtiment;
8. montrer les métriques si vérité terrain disponible;
9. montrer l'incertitude;
10. montrer l'export PNG/JSON.

Message clé :

> La démo n'est pas là pour montrer une IA magique, mais pour montrer une chaîne complète d'aide à la décision.

### Slide 10 - Résultats quantitatifs

Durée cible : 1 minute 30.

Comparer :

- baseline U-Net + TTA d4;
- champion Siamese Attention;
- pipeline avec segmentation bâtiment;
- borne haute oracle building si utile.

Résultats à mettre en avant :

- U-Net + TTA d4 : F1 damaged = 0.631300, IoU damaged = 0.461240, mean IoU = 0.681574;
- nouveau champion damage : F1 damaged = 0.701317, IoU damaged = 0.540022, mean IoU = 0.728266;
- champion building : F1 building = 0.850421, IoU building = 0.739767.

### Slide 11 - Explicabilité visuelle

Durée cible : 1 minute.

Montrer :

- image avant;
- image après;
- damage brut;
- masque bâtiment;
- damage final post-processé;
- overlay.

Message : l'explicabilité n'est pas seulement une courbe; ici, elle est visuelle et lisible par un utilisateur métier.

### Slide 12 - Modèle d'affaires

Durée cible : 1 minute.

Présenter Aftermath comme un produit B2B/B2G :

- SaaS pour organisations;
- licence annuelle;
- usage par crise;
- API ou intégration SIG;
- accompagnement d'équipes humanitaires ou institutionnelles.

### Slide 13 - Limites, éthique et supervision humaine

Durée cible : 1 minute.

Points :

- dépendance aux images satellite;
- erreurs possibles de segmentation;
- modèle actuel en 3 classes, pas encore score officiel 5 classes xView2;
- besoin de supervision humaine;
- transparence sur les incertitudes.

### Slide 14 - Roadmap et conclusion

Durée cible : 1 minute.

Roadmap :

- améliorer encore le modèle damage;
- analyser les architectures avancées;
- passer vers 5 classes xView2;
- intégrer SIG/API;
- améliorer l'expérience produit.

Conclusion :

> Aftermath montre qu'il est possible de passer d'un modèle de segmentation à un prototype produit capable d'aider à prioriser l'analyse post-catastrophe.

## Répartition temporelle recommandée

| Partie | Durée |
| --- | ---: |
| Problème, utilisateurs, solution | 3 minutes |
| Données et pipeline IA | 2 minutes |
| Démonstration centrale | 4 à 5 minutes |
| Résultats et explicabilité | 3 minutes |
| Business model, limites, roadmap | 2 à 3 minutes |

Total : environ 15 minutes.

## Consigne pour NotebookLM

Produire une présentation claire, visuelle et orale. Éviter un deck trop technique ou trop marketing. Le bon équilibre est : **un produit crédible, soutenu par une vraie preuve IA**.
