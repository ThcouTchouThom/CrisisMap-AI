# Problème et solution

## Contexte

Les catastrophes naturelles provoquent souvent des dégâts étendus sur des zones difficiles d'accès. Après un ouragan, un incendie, un séisme, une inondation ou un tsunami, il faut répondre vite à des questions critiques :

- quelles zones sont les plus touchées ;
- quels bâtiments semblent encore utilisables ;
- où envoyer les équipes de secours en priorité ;
- quels quartiers nécessitent une inspection détaillée ;
- comment documenter l'ampleur des dommages.

Les images satellite sont très utiles, car elles couvrent rapidement de larges zones. Mais elles doivent encore être interprétées. Cette interprétation peut devenir lente si elle est faite uniquement à la main.

## Utilisateurs cibles

Aftermath peut intéresser plusieurs types d'utilisateurs :

- ONG et organisations humanitaires ;
- collectivités locales ;
- cellules de crise ;
- équipes SIG ;
- assureurs ;
- organismes de gestion des risques ;
- équipes de recherche en télédétection.

Le point commun entre ces acteurs est le besoin d'une première évaluation rapide, même imparfaite, pour orienter les décisions.

## Problème opérationnel

Le problème n'est pas seulement de classifier une image. Il faut comparer deux moments :

- avant catastrophe : état normal de la zone ;
- après catastrophe : état potentiellement dégradé.

La difficulté est donc multi-temporelle. Le modèle doit comprendre ce qui a changé et déterminer si ce changement correspond à un bâtiment endommagé.

## Solution proposée

Aftermath propose un pipeline d'IA capable de :

1. recevoir une paire d'images satellite pré/post catastrophe ;
2. produire un masque de segmentation damage ;
3. distinguer fond, bâtiment intact et bâtiment endommagé ;
4. utiliser un modèle building pour mieux localiser les bâtiments ;
5. afficher le résultat dans une interface Streamlit claire.

## Pourquoi la segmentation

La segmentation est plus adaptée qu'une simple classification d'image, car la question n'est pas seulement : "y a-t-il des dégâts ?" mais plutôt : "où sont les dégâts ?"

Le résultat attendu est une carte pixel par pixel :

- noir : fond ;
- vert : bâtiment intact ;
- rouge : bâtiment endommagé.

Cette sortie est visuelle, interprétable et directement exploitable pour une démonstration.

## Dataset utilisé

Le projet utilise **xBD / xView2**, un dataset de référence pour l'évaluation des dommages sur bâtiments après catastrophe.

Il contient :

- des images pré-catastrophe ;
- des images post-catastrophe ;
- des annotations de bâtiments ;
- des niveaux de dégâts.

Dans la version actuelle du projet, les dégâts sont simplifiés en 3 classes :

- fond ;
- bâtiment intact ;
- bâtiment endommagé.

Cette simplification permet de consolider le pipeline avant de revenir à une formulation plus fine proche du score officiel xView2.

## Valeur produit

Aftermath peut apporter une valeur concrète :

- accélérer une première analyse visuelle ;
- réduire la charge de tri manuel ;
- fournir une carte de priorisation ;
- aider à communiquer l'état d'une zone ;
- préparer une inspection terrain plus ciblée.

Le système ne prétend pas donner une décision finale. Il fournit une couche d'aide à l'analyse.

## Message pour l'oral

Le projet part d'un besoin réel : après une catastrophe, il faut comprendre vite où sont les dégâts. Notre solution utilise des images satellite avant/après et un pipeline de segmentation pour générer une carte lisible des bâtiments intacts et endommagés.
