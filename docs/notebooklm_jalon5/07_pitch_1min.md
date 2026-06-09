# Pitch 1 minute

## Version orale

Aftermath aide à répondre à une question critique après une catastrophe : où sont les dégâts et où faut-il agir en priorité ?

Notre solution analyse des paires d'images satellite avant et après catastrophe. Elle produit une carte visuelle qui distingue le fond, les bâtiments intacts et les bâtiments endommagés. L'objectif n'est pas de remplacer les experts terrain, mais de leur fournir une première lecture rapide et structurée.

Techniquement, nous utilisons le dataset xBD / xView2. Nous avons commencé par une baseline U-Net forte, puis nous avons amélioré le modèle avec une architecture Siamese Attention adaptée au pré/post. Le nouveau champion atteint un F1 damaged de 0.7013, contre 0.6313 pour la baseline U-Net avec TTA d4. Nous avons aussi un modèle de segmentation bâtiment U-Net++ EfficientNet-B4 avec un F1 building de 0.8504.

Le prototype Streamlit permet de téléverser une paire d'images ou d'utiliser un exemple du dataset, de lancer l'inférence, puis de visualiser le damage brut, le masque bâtiment et la carte finale post-processée.

La valeur produit est claire : accélérer l'analyse post-catastrophe, aider à prioriser les interventions et fournir une estimation visuelle rapide aux ONG, collectivités, assureurs et cellules de crise.

## Version très courte

Aftermath transforme des images satellite avant/après catastrophe en cartes de dégâts. Notre meilleur modèle Siamese Attention atteint un F1 damaged de 0.7013, et notre modèle building atteint un F1 de 0.8504. Le prototype Streamlit permet une démonstration complète : upload d'une paire réelle, inférence, masque bâtiment, damage final et overlay visuel. L'objectif est d'aider les équipes de crise à voir les dégâts plus vite pour agir plus vite.

## Formule de valeur

**Après une catastrophe, Aftermath fournit une première carte visuelle des bâtiments endommagés afin d'accélérer la priorisation des interventions.**

## Utilisateurs cibles

- ONG ;
- collectivités ;
- cellules de crise ;
- équipes SIG ;
- assureurs ;
- organismes de gestion des risques.

## Différenciation

Aftermath combine :

- analyse pré/post satellite ;
- segmentation damage ;
- segmentation building ;
- post-processing explicable ;
- interface interactive ;
- visualisation directe des incertitudes et des étapes intermédiaires.
