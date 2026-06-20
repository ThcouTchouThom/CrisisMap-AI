# Storyboard de démonstration live finale

Ce document prépare une démonstration centrale de 4 à 5 minutes pour la présentation finale du projet Aftermath / CrisisMap AI.

La démo doit utiliser l'application classique, qui est la version finale retenue :

```powershell
streamlit run app/streamlit_app.py
```

Le plan B consiste à utiliser les exemples embarqués ou des captures déjà générées si l'inférence live est lente.


Phrase obligatoire à intégrer pendant la démo :

> La démo n'est pas là pour montrer une IA magique, mais pour montrer une chaîne complète d'aide à la décision.

## Objectif de la démo

Montrer que le prototype est capable de :

- recevoir une paire d'images satellite pré/post catastrophe;
- lancer une inférence IA;
- produire une carte visuelle des dégâts;
- afficher les étapes intermédiaires;
- expliquer le rôle du masque bâtiment;
- fournir des métriques quand une vérité terrain existe;
- exporter des résultats.

La démo doit être fluide, visuelle et compréhensible par une personne qui ne connaît pas les détails du deep learning.

## Durée cible

Durée : 4 à 5 minutes.

Répartition :

| Étape | Durée |
| --- | ---: |
| Ouverture et rappel du but | 30 s |
| Choix du pipeline et de la paire | 45 s |
| Inférence et overlay final | 1 min |
| Sorties intermédiaires et masque bâtiment | 1 min 30 |
| Métriques, incertitude, export | 1 min |

## Préparation avant la présentation

Avant d'entrer en salle :

- vérifier que les checkpoints existent localement;
- vérifier que l'application classique se lance;
- précharger une ou deux paires visuellement bonnes;
- garder l'application sûre prête dans un autre terminal;
- garder une capture ou une planche de contact en secours;
- éviter de dépendre d'un téléchargement ou d'une ressource réseau.

Prévoir une paire dataset recommandée et une paire upload si possible.

## Déroulé détaillé

### 1. Ouverture de l'application

Action :

- ouvrir l'application classique;
- montrer le header Aftermath;
- rappeler le slogan : **Voir les dégâts pour agir plus vite**.

Phrase possible :

> Voici le prototype Aftermath. L'idée est de partir d'une image satellite avant catastrophe et d'une image après catastrophe, puis de produire une carte lisible des bâtiments intacts et endommagés.

### 2. Choix du pipeline qualité maximale

Action :

- dans la sidebar, sélectionner le pipeline :
  **Qualité maximale : damage + TTA d4 + building post-process**.

Phrase possible :

> Pour la démonstration, j'utilise le pipeline le plus complet : le modèle damage, une stabilisation par TTA d4, puis une segmentation bâtiment qui aide à rendre la carte plus cohérente.

À préciser :

- le mode rapide existe pour aller vite;
- le mode qualité utilise TTA d4;
- le mode qualité maximale ajoute le masque bâtiment et le component majority.

### 3. Chargement d'une paire satellite

Action :

- choisir un exemple dataset si disponible;
- sinon téléverser une paire pré/post préparée.

Phrase possible :

> Ici, on charge une paire réelle du dataset xBD : la scène avant catastrophe et la même zone après catastrophe.

### 4. Lancement de l'inférence

Action :

- cliquer sur le bouton d'analyse;
- laisser l'application afficher le résultat.

Phrase possible pendant l'attente :

> Le modèle compare les deux images, prédit les classes de damage, puis applique le post-traitement bâtiment pour produire la carte finale.

Si l'inférence prend quelques secondes :

> C'est une inférence en 1024 pixels avec plusieurs passes de TTA, donc elle est plus lourde que le mode rapide, mais elle donne une sortie plus stable pour la démo.

### 5. Lecture de l'overlay final

Action :

- montrer l'overlay final sur l'image post-catastrophe;
- expliquer les couleurs.

Phrase possible :

> Le résultat principal est cet overlay : le vert indique les bâtiments détectés comme intacts, le rouge les bâtiments endommagés. L'objectif est que cette carte soit lisible rapidement par une cellule de crise.

Message important :

- l'overlay n'est pas seulement une sortie technique;
- c'est le support visuel principal pour l'utilisateur.

### 6. Passage dans les sorties intermédiaires

Action :

- ouvrir l'onglet **Model output** ou équivalent;
- montrer :
  - image avant;
  - image après;
  - damage brut;
  - masque bâtiment;
  - damage final.

Phrase possible :

> On ne cache pas le pipeline. On peut voir le damage brut, le masque bâtiment, puis le résultat final post-processé. Cela permet de comprendre pourquoi certains pixels sont conservés ou supprimés.

### 7. Explication du masque bâtiment

Action :

- montrer le masque bâtiment;
- expliquer son rôle dans le post-processing.

Phrase possible :

> Le masque bâtiment sert à contraindre la prédiction damage. En pratique, on veut éviter de prédire des dégâts sur du fond, une route ou une zone non bâtie. Le post-processing applique ensuite une décision plus cohérente par composante bâtiment.

Nuance importante :

> Ce masque améliore les résultats, mais il peut aussi supprimer de vrais positifs si le segmentateur bâtiment rate un bâtiment. C'est pour cela que nous gardons les sorties intermédiaires visibles.

### 8. Onglet métriques si vérité terrain disponible

Action :

- ouvrir l'onglet métriques si le mode dataset fournit une vérité terrain.

Phrase possible :

> Quand la vérité terrain est disponible, on peut calculer les métriques : F1 damaged, IoU damaged et mean IoU. Ce sont les métriques qui nous permettent de comparer les versions du pipeline.

Rappel des résultats principaux :

- baseline U-Net + TTA d4 : F1 damaged = 0.631300;
- champion actuel : F1 damaged = 0.701317;
- champion building : F1 building = 0.850421.

### 9. Onglet incertitude

Action :

- ouvrir l'onglet incertitude;
- montrer l'entropie ou les cartes de probabilité.

Phrase possible :

> L'incertitude est importante : dans une vraie situation, il faut savoir où le modèle est confiant et où il hésite. Cela aide à décider quelles zones doivent être vérifiées en priorité.

### 10. Export PNG / JSON

Action :

- montrer les options d'export;
- ne pas forcément télécharger si le temps est court.

Phrase possible :

> Enfin, l'application permet d'exporter un masque, un overlay ou un petit rapport JSON. C'est le début d'un workflow exploitable, par exemple vers une équipe SIG ou une cellule de crise.

## Phrase de conclusion de la démo

> Cette démonstration montre la chaîne complète : image satellite, modèle IA, stabilisation, masque bâtiment, carte finale, métriques et export. L'objectif n'est pas de remplacer les analystes, mais de leur donner une première carte rapide et explicable.

## Plan B si l'inférence est lente

Si l'inférence est trop lente :

1. passer au pipeline **Qualité : damage + TTA d4** ou **Rapide : damage seul**;
2. utiliser une paire déjà analysée dans l'historique;
3. afficher une capture ou une planche de contact préparée;
4. expliquer que le pipeline qualité maximale est plus coûteux parce qu'il fait plusieurs passes de modèle.

Phrase possible :

> Pour respecter le temps de présentation, je passe sur une sortie déjà générée. Le principe reste le même : on conserve le pipeline complet, mais on évite d'attendre l'inférence en direct.

## Plan B si l'exemple visuel est moins bon que prévu

Si la prédiction est peu lisible :

1. changer rapidement de paire dataset;
2. utiliser une paire présélectionnée pour la démo;
3. montrer les sorties intermédiaires pour expliquer l'erreur;
4. transformer l'exemple en discussion sur les limites.

Phrase possible :

> Cet exemple illustre aussi une limite importante : le modèle peut se tromper. C'est pour cela que nous affichons le masque building, le damage brut, le résultat final et l'incertitude. L'application est conçue pour aider l'analyse, pas pour masquer les erreurs.

## Ce qu'il ne faut pas faire

- ne pas passer trop de temps dans les détails d'architecture pendant la démo;
- ne pas prétendre que le modèle est parfait;
- ne pas présenter l'overlay comme une vérité terrain;
- ne pas ouvrir trop d'onglets ou de fichiers externes;
- ne pas lancer une longue inférence sans plan B.

## Message final pour NotebookLM

La démo doit être racontée comme une expérience utilisateur :

1. je charge une scène;
2. je choisis un niveau de qualité;
3. je lance l'analyse;
4. je lis une carte;
5. je comprends les étapes;
6. j'exporte un résultat.

Le ton doit être confiant, mais responsable.
