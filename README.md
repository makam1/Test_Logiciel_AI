Données

Ce dossier accueille le jeu de données Credit Card Fraud Detection (Kaggle). 

Contenu attendu

Fichier : `creditcard.csv`
284 807 transactions de cartes bancaires européennes (septembre 2013, deux jours)
492 fraudes, soit 0,17 % : classes très déséquilibrées
Colonnes : `Time`, `V1` à `V28` (composantes PCA anonymisées), `Amount`, `Class` (0 = normale, 1 = fraude)

Mise en place

Télécharger le jeu depuis Kaggle (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download)
Placer `creditcard.csv` dans ce dossier (`data/creditcard.csv`).
Ne jamais le committer : `.gitignore` l'exclut

Limites à connaître

Données de 2013 : les comportements de fraude ont évolué.
Variables anonymisées : aucun attribut sur les personnes, donc pas d'audit d'équité possible sur des groupes protégés.
Sans les données, les tests qui en dépendent sont ignorés (cas de la CI GitHub).
