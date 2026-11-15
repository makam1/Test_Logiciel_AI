# Preuves d'intégration continue

- Dépôt : https://github.com/makam1/Test_Logiciel_AI
- Workflow : `.github/workflows/tests.yml`

## 1. Ce que fait le workflow

| Déclencheur | Usage |
|---|---|
| Pull Request | Contrôle rapide à chaque proposition de modification |
| Push sur `main` | Contrôle après fusion |
| Manuel (`workflow_dispatch`) | Relance à la demande |
| Hebdomadaire (lundi 06:00 UTC) | Détecte une régression due à une nouvelle version d'une bibliothèque non figée |

Étapes : récupération du code, installation de Python 3.14 et des dépendances, vérification du style avec Ruff, exécution de pytest avec couverture (échec sous 85 %), conservation des rapports (couverture HTML et XML, JUnit) même en cas d'échec.

## 2. Preuve 1 : exécution réussie

- Pull Request : https://github.com/makam1/Test_Logiciel_AI/pull/1
- Lien du run : https://github.com/makam1/Test_Logiciel_AI/actions/runs/43758145861
- Résultat : toutes les étapes réussies en 44 secondes ; **74 tests passés, 1 ignoré** (non-régression sur les vraies données, absentes du dépôt) ; couverture **94,97 %** (159 instructions, 8 non couvertes), seuil exigé : 85 %.

La première tentative avait échoué avant de démarrer : « The job was not started because your account is locked due to a billing issue » (voir A07). Après règlement du problème de paiement, la relance du même run a réussi, sans modification du code.

## 3. Preuve 2 : échec volontaire correctement détecté

- Modification : dans `app/evaluation.py`, le taux de faux positifs `fp / (fp + tn)` a été remplacé par `fp / tn` (commit `d36575e`, branche `ci/echec-volontaire`).
- Détection locale : `pytest` signale 1 test en échec
- Pull Request fermée sans fusion : https://github.com/makam1/Test_Logiciel_AI/pull/2
- Résultat : Ruff réussit, puis l'étape de tests échoue : `FAILED tests/test_evaluation.py::test_confusion_counts_and_rates` (obtenu 0,3333, attendu 0,25), **1 échec, 73 réussis, 1 ignoré**, `Process completed with exit code 1`. GitHub affiche « All checks have failed ». Les rapports de test sont tout de même conservés.
- La branche a été supprimée sans jamais être fusionnée : `main` n'a pas reçu le code défectueux.


## 4. Limites

- Le jeu de données n'est pas versionné : sur GitHub, le test de non-régression sur les vraies données est ignoré. La CI vérifie la logique de décision sur données synthétiques et la cohérence du fichier de baseline ; la comparaison complète s'exécute en local.
- Les entraînements complets et les tests de charge ne sont pas dans la CI : ils sont trop coûteux pour chaque Pull Request et nécessitent des données absentes du dépôt.
- Les dépendances ne sont pas figées : la CI a utilisé Python 3.14.10 alors que les mesures locales datent de 3.14.4. L'exécution hebdomadaire sert à détecter une casse due à une mise à jour.
- Lignes non couvertes (8) : `app/evaluation.py` ligne 41, `app/regression.py` lignes 31 à 38 (probablement la fonction qui lit le hash Git, utilisée seulement par les scripts), `app/training.py` ligne 43.
- Une première série d'exécutions rouges apparaît dans l'historique avant le déblocage du compte : elle ne reflète aucune défaillance du code.
