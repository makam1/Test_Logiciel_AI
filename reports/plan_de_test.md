# Plan de test - AI-TestGuard

Qualification d'un modèle de détection de fraude bancaire (Credit Card Fraud Detection, Kaggle).

## 1. Périmètre

**Inclus dans les tests :**
- Données : schéma, types, valeurs manquantes ou infinies, cible binaire
- Prétraitement : mise à l'échelle, cohérence des dimensions
- Modèle : entraînement (régression logistique, arbre de décision), probabilités produites
- Prédiction : sortie valide, rechargement du pipeline sauvegardé
- Intégration : chaîne complète chargement -> séparation -> entraînement -> sauvegarde -> prédiction -> métriques
- Robustesse, non-régression, performance d'inférence, CI GitHub Actions

**Exclus :**
- Déploiement réel en production bancaire
- Analyse d'équité selon l'âge, le sexe ou la localisation (variables absentes du jeu de données)
- Validation réglementaire et métier (hors périmètre technique)

## 2. Exigences et matrice de test

| ID | Exigence | Type de test | Risque maîtrisé | Résultat attendu | Test pytest associé |
|---|---|---|---|---|---|
| T01 | Les colonnes nécessaires sont présentes | Données | Pipeline qui plante ou prédit sur des données incomplètes | Aucune colonne indispensable manquante | tests/test_preprocessing.py::test_required_columns |
| T02 | Les données respectent les types attendus | Unitaire | Erreurs silencieuses dues à un mauvais typage | Types conformes au contrat | tests/test_preprocessing.py::test_column_types |
| T03 | Le prétraitement ne produit pas de valeurs non finies | Unitaire | NaN ou infinis qui faussent l'entraînement | Aucune valeur NaN ou infinie en sortie | tests/test_preprocessing.py::test_no_non_finite |
| T04 | Le modèle produit une probabilité valide | Fonctionnel ML | Scores incohérents, seuil inexploitable | Probabilité comprise entre 0 et 1 | tests/test_prediction.py::test_probabilities_valid |
| T05 | Les performances sont évaluées sur des données indépendantes | ML | Surestimation des performances | Jeu de test non utilisé pour le réglage | tests/test_integration.py::test_split_no_overlap |
| T06 | Les résultats sont reproductibles | Non-régression | Résultats différents à chaque exécution | Écart inférieur à la tolérance définie | tests/test_regression.py::test_reproducibility |
| T07 | Les tests s'exécutent sur GitHub | CI | Code cassé fusionné sans alerte | Workflow réussi ou échec correctement signalé | .github/workflows/tests.yml |
| T08 | Le scaler est ajusté uniquement sur les données d'entraînement | Intégration | Fuite de données, métriques artificiellement bonnes | fit jamais appelé sur validation ou test | tests/test_integration.py::test_scaler_fit_on_train_only |
| T09 | La cible ne contient que 0 et 1, sans colonne dupliquée | Données | Entraînement sur une cible corrompue | Rejet explicite si valeur autre que 0 ou 1 | tests/test_preprocessing.py::test_target_binary |
| T10 | La proportion de fraudes est conservée dans train, validation et test | Intégration | Jeu de test sans fraude, métriques non calculables | Écart de proportion inférieur à 0,1 point | tests/test_integration.py::test_stratified_split |
| T11 | Le pipeline sauvegardé puis rechargé donne les mêmes prédictions | Intégration | Modèle déployé différent du modèle validé | Prédictions identiques à 1e-9 près | tests/test_integration.py::test_save_reload_same_predictions |
| T12 | Une entrée invalide est rejetée avec une erreur explicite | Robustesse | Prédiction silencieuse sur des données absurdes | ValueError avec message clair | tests/test_robustness.py::test_invalid_input_rejected |
| T13 | Le rappel sur la validation dépasse le seuil minimal | ML | Trop de fraudes non détectées | Rappel >= 0,80 sur la validation | tests/test_regression.py::test_recall_threshold |
| T14 | La latence P95 d'inférence reste sous le seuil | Performance | Modèle trop lent pour un usage temps réel | P95 < 50 ms par transaction | tests/test_robustness.py::test_latency_p95 |

## 3. Risques principaux

| Risque | Gravité | Exigences associées |
|---|---|---|
| Fraudes non détectées (faux négatifs) : pertes financières | Élevée | T04, T13 |
| Fausses alertes (faux positifs) : paiements bloqués, clients mécontents | Moyenne | T04, T13 |
| Fuite de données : performances annoncées irréalistes | Élevée | T05, T08, T10 |
| Données invalides acceptées silencieusement | Élevée | T01, T02, T03, T09, T12 |
| Régression non détectée après modification | Moyenne | T06, T07, T11 |
| Modèle trop lent pour la production | Moyenne | T14 |

## 4. Outils et preuves conservées

| Outil | Usage | Preuve conservée |
|---|---|---|
| pytest | Tests unitaires, intégration, robustesse | Rapport d'exécution dans reports/ |
| pytest-cov | Couverture de code | Rapport de couverture (terminal et HTML) |
| Ruff | Vérification du style | Sortie de `ruff check` |
| GitHub Actions | Exécution automatique des tests | Captures d'une exécution réussie et d'un échec volontaire |
| scikit-learn, joblib | Pipeline, modèles, sauvegarde | Fichier de baseline des métriques |
| Hypothesis, psutil | Cas générés, mesures de ressources | Tableaux de résultats dans reports/ |

## 5. Critères d'acceptation et de rejet

Les seuils sont fixés à partir du contexte métier et réglés sur les données d'entraînement et de validation, jamais sur le jeu de test.

**Le modèle est accepté pour une phase pilote si :**
- tous les tests critiques (T01 à T12) passent ;
- le rappel sur la validation est >= 0,80 (une fraude non détectée coûte plus cher qu'une fausse alerte) ;
- le taux de faux positifs est <= 1 % (au-delà, trop de clients légitimes sont bloqués) ;
- la PR-AUC est supérieure à celle d'un modèle de référence naïf ;
- la latence P95 est < 50 ms par transaction ;
- aucune régression au-delà de la tolérance définie par rapport à la baseline ;
- le workflow GitHub Actions est vert.

**Le modèle est rejeté ou renvoyé en investigation si :**
- un test critique échoue ;
- une fuite de données est détectée ;
- le rappel ou le taux de faux positifs sort des seuils ;
- la baseline est dégradée au-delà de la tolérance.

**Limite :** réussir ces tests techniques ne suffit pas à autoriser un déploiement. Il faut aussi une validation métier, opérationnelle, sécuritaire et réglementaire.
