# Consignes pour le deck final

Ce fichier donne les règles à suivre pour générer le dernier deck de présentation Aftermath / CrisisMap AI.

## Format général

La présentation finale doit durer environ **15 minutes**, suivies de questions.

Elle doit être structurée comme un **pitch de produit technologique** :

- claire pour des personnes non spécialistes;
- crédible techniquement;
- centrée sur une démonstration;
- honnête sur les limites;
- orientée usage terrain.

Le ton doit être sérieux, produit, terrain et crédible. Il ne faut pas donner l'impression d'un projet purement académique, ni d'un produit déjà industrialisé au-delà de ce qui existe réellement.

## Nombre de slides

Maximum recommandé : **12 slides principales**.

Il peut y avoir quelques slides de backup pour les questions, mais le deck principal doit rester court.

Structure possible :

1. Aftermath : voir les dégâts pour agir plus vite
2. Problème terrain et utilisateurs
3. Produit proposé
4. Données xBD / xView2 et formulation IA
5. Pipeline actuel
6. Démonstration live
7. Résultats damage
8. Segmentation bâtiment et post-processing
9. Explicabilité visuelle
10. Modèle d'affaires
11. Limites et éthique
12. Roadmap et conclusion

## Démo centrale

La démonstration doit occuper **4 à 5 minutes**.

Elle doit être au centre de la présentation, pas seulement en annexe.

À montrer :

- ouverture de l'application classique;
- sélection du pipeline qualité maximale;
- choix ou upload d'une paire satellite;
- lancement de l'inférence;
- lecture de l'overlay final;
- affichage des sorties intermédiaires;
- explication du masque bâtiment;
- métriques si vérité terrain disponible;
- incertitude;
- exports PNG/JSON.

Phrase à intégrer :

> La démo n'est pas là pour montrer une IA magique, mais pour montrer une chaîne complète d'aide à la décision.

## Équilibre pitch produit / preuve technique

Le deck doit garder deux niveaux :

### Niveau produit

- à qui sert Aftermath;
- quelle douleur terrain il résout;
- comment l'interface est utilisée;
- pourquoi cela peut créer de la valeur;
- quelles sont les limites d'un usage réel.

### Niveau technique

- dataset xBD / xView2;
- paires pré/post catastrophe;
- segmentation 3 classes;
- Siamese Attention;
- TTA d4;
- U-Net++ EfficientNet-B4 pour building;
- component majority;
- métriques F1 damaged, IoU damaged, mean IoU.

## Ce qu'il ne faut pas dire

Ne pas dire que le prototype fait déjà :

- export SIG complet;
- GeoJSON;
- GeoTIFF;
- intégration QGIS;
- intégration ArcGIS;
- API publique;
- géoréférencement opérationnel.

Ces éléments doivent être mentionnés uniquement dans la roadmap.

Formulations correctes :

- "Ces exports SIG font partie des prochaines étapes."
- "L'intégration QGIS/ArcGIS est une perspective produit."
- "Le géoréférencement opérationnel est nécessaire pour une version terrain."

Formulations à éviter :

- "Aftermath exporte déjà du GeoJSON."
- "Le prototype est intégré à ArcGIS."
- "Nous avons déjà un pipeline SIG complet."

## Explicabilité

Éviter les formulations trop absolues :

- "explicabilité totale";
- "modèle parfaitement interprétable";
- "raisonnement complet de l'IA".

Préférer :

- "explicabilité visuelle";
- "traçabilité du raisonnement";
- "visualisation des étapes intermédiaires";
- "lecture du masque bâtiment et du damage final";
- "aide à comprendre la prédiction".

Message :

> L'application rend le résultat plus transparent en montrant l'image avant, l'image après, le damage brut, le masque bâtiment, le damage post-processé et l'overlay final.

## Résultats à citer

Résultats damage :

- U-Net + TTA d4 : F1 damaged = 0.6313, IoU damaged = 0.4612, mean IoU = 0.6816;
- champion intégré : `dftv2_hist1000_attention_sqrt2_ft_250_seed0`, F1 damaged = 0.7013, IoU damaged = 0.5400, mean IoU = 0.7283;
- dernier run marginalement meilleur : `dftv2_hist1000_attention_sqrt4_ft_400_seed0`, F1 damaged = 0.7018, IoU damaged = 0.5406, mean IoU = 0.7273.

Résultats building :

- `b400_effb4_sampler8_ft`;
- U-Net++ EfficientNet-B4;
- F1 building = 0.8504;
- IoU building = 0.7398.

Décision importante :

> Le dernier run damage est très légèrement meilleur, mais le modèle intégré reste celui déjà stabilisé, portable et testé dans l'application.

## Ton recommandé

Le deck doit donner l'impression suivante :

- le problème est réel;
- le prototype est démontrable;
- les résultats sont sérieux;
- l'équipe sait ce qui est fait et ce qui ne l'est pas encore;
- la roadmap est crédible;
- l'usage reste responsable.

## Message final

Conclusion à faire ressortir :

> Aftermath n'est pas seulement un modèle de segmentation. C'est le début d'un produit d'aide à la décision : une interface qui transforme des images satellite pré/post catastrophe en une carte visuelle, explicable et exploitable pour prioriser l'analyse terrain.
