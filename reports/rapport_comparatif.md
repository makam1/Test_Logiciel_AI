# Rapport de non-régression : référence contre nouvelle version

## 1. Principe

Une baseline est un instantané figé de la version validée. Elle est enregistrée dans `reports/baseline.json` par `make_baseline.py` et contient : la version du code (hash du commit), les paramètres du modèle, la graine aléatoire, le schéma de séparation des données, le seuil de décision, les métriques sur le jeu de validation, les versions de Python et des bibliothèques, et les caractéristiques du jeu de données (nombre de lignes, de fraudes, empreinte SHA-256).

La baseline n'est jamais recalculée automatiquement : `make_baseline.py` refuse d'écraser le fichier sans l'option `--force`. Un recalcul automatique à chaque version masquerait les régressions. La baseline a été régénérée une fois avec `--force`, car la première avait été créée depuis un répertoire de travail comportant des modifications non commitées (traçabilité insuffisante).

## 2. Baseline (validation, régression logistique, seuil 0,9991)

| Métrique | Valeur |
|---|---|
| Précision | 0,6991 |
| Rappel | 0,8061 |
| F1 | 0,7488 |
| PR-AUC | 0,6857 |
| Taux de faux positifs | 0,0006 |

## 3. Tolérances

Elles sont fixées une fois pour toutes, avant toute comparaison. Le jeu de validation contient 98 fraudes et 56 863 transactions normales.

| Métrique | Dégradation tolérée | Justification |
|---|---|---|
| Rappel | 0,03 | 1 fraude vaut environ 0,01 de rappel : 0,03 correspond à 3 fraudes manquées |
| Précision | 0,05 | Variation d'ordre statistique liée au faible nombre de fraudes |
| PR-AUC | 0,02 | Métrique globale plus stable, donc tolérance plus stricte |
| FPR | 0,0003 | Environ 17 fausses alertes supplémentaires sur 56 863 transactions normales |

## 4. Règles de décision

| Décision | Condition |
|---|---|
| REJETER | Rappel ou PR-AUC dégradé au-delà de la tolérance, ou critère d'acceptation violé (rappel < 0,80 ou FPR > 1 %) |
| INVESTIGUER | Précision ou FPR dégradé au-delà de la tolérance, ou gain supérieur à 0,05 sur une métrique (risque de fuite de données) |
| ACCEPTER | Aucune dégradation au-delà des tolérances |

Le script `check_regression.py` retourne le code 0 (accepter), 1 (rejeter) ou 2 (investiguer), ce qui permet de l'utiliser dans un contrôle automatisé.

## 5. Comparaison 1 : version identique à la référence

| Métrique | Baseline | Version évaluée | Dégradation | Tolérance | Verdict |
|---|---|---|---|---|---|
| Rappel | 0,8061 | 0,8061 | 0,0000 | 0,03 | OK |
| Précision | 0,6991 | 0,6991 | 0,0000 | 0,05 | OK |
| PR-AUC | 0,6857 | 0,6857 | 0,0000 | 0,02 | OK |
| FPR | 0,0006 | 0,0006 | 0,0000 | 0,0003 | OK |

**Décision : ACCEPTER** (code retour 0). Ce résultat démontre aussi la reproductibilité : le réentraînement donne exactement les mêmes métriques.

## 6. Comparaison 2 : modification simulée (arbre de décision à la place de la régression logistique)

| Métrique | Baseline | Version évaluée | Dégradation | Tolérance | Verdict |
|---|---|---|---|---|---|
| Rappel | 0,8061 | 0,8061 | 0,0000 | 0,03 | OK |
| Précision | 0,6991 | 0,3015 | +0,3976 | 0,05 | Hors tolérance |
| PR-AUC | 0,6857 | 0,4891 | +0,1966 | 0,02 | Hors tolérance |
| FPR | 0,0006 | 0,0032 | +0,0026 | 0,0003 | Hors tolérance |

**Décision : REJETER** (code retour 1).

**Motivation.** À rappel identique, la précision est divisée par plus de deux et le modèle produit environ cinq fois plus de fausses alertes (183 contre 34 sur la validation). La PR-AUC, qui est une métrique critique, chute de 0,20. Le critère de rejet est donc déclenché : en production, cette modification bloquerait de nombreux paiements légitimes sans détecter davantage de fraudes.

## 7. Limites

- **Le rappel est épinglé par construction** : le seuil est re-choisi sur la validation pour atteindre un rappel de 0,80, donc le rappel varie très peu d'une version à l'autre. Les régressions se voient dans la précision, le FPR et la PR-AUC. Une alternative serait de figer le seuil de la baseline et de mesurer l'évolution du rappel à seuil constant.
- La baseline repose sur une seule séparation des données et 98 fraudes en validation : une tolérance de 0,03 sur le rappel représente seulement 3 transactions.
- Les résultats dépendent de la version des bibliothèques et du jeu de données exact : leurs versions et l'empreinte SHA-256 sont enregistrées pour détecter un changement.
- Le test de non-régression sur les vraies données est ignoré sur GitHub, car le jeu de données n'est pas versionné. La CI vérifie la logique de décision sur des données synthétiques et la cohérence du fichier de baseline ; la comparaison complète s'exécute en local ou manuellement.
- Les tolérances sont des choix de l'ingénieur QA, justifiés mais pas validés par la banque.

## 8. Conclusion

La version actuelle du pipeline est identique à la référence : **ACCEPTER**. Une modification dégradant la précision et la PR-AUC au-delà des tolérances est détectée et **REJETÉE** automatiquement. Le mécanisme est reproductible (mêmes graines, mêmes données, mêmes versions) et traçable (hash du commit, versions des bibliothèques et empreinte du jeu de données dans la baseline).
