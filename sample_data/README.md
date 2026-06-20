# Données de démonstration embarquées

Ce dossier contient quelques paires d'images satellite pré/post catastrophe sélectionnées pour tester le prototype Aftermath sans télécharger le dataset xBD complet.

Ces fichiers servent uniquement à la démonstration de l'application Streamlit :

```powershell
python -m streamlit run app/streamlit_app.py
```

Dans l'application, utiliser le mode **Exemples inclus**.

## Contenu

Chaque sous-dossier de `demo_pairs/` contient au minimum :

- `pre.png` : image avant catastrophe;
- `post.png` : image après catastrophe;
- `README.md` : description rapide.

Quand disponibles :

- `target.png` : masque de vérité terrain colorisé;
- `overlay_target.png` : vérité terrain superposée à l'image post-catastrophe.

Ces exemples ne remplacent pas le dataset xBD/xView2 complet. Ils sont fournis pour rendre le dépôt GitHub testable immédiatement.
