# Stratégie de surveillance après déploiement

## 1. Objectif

Détecter rapidement qu'un modèle déployé se dégrade ou qu'il reçoit des données différentes de celles de son entraînement, en tenant compte du fait que les vraies étiquettes (fraude confirmée ou non) arrivent avec du retard.

## 2. Indicateurs surveillés

| Indicateur | Pourquoi | Disponible | Seuil et alerte |
|---|---|---|---|
| Distribution de chaque variable (PSI, indice de stabilité de la population) | Détecte la dérive des données | Immédiatement, sur chaque lot | PSI >= 0,10 : avertissement ; PSI >= 0,25 : alerte |
| Part de transactions hors de la plage d'entraînement | Détecte les valeurs aberrantes ou une rupture de format | Immédiatement | Alerte au-delà de 1 % |
| Taux de transactions signalées comme frauduleuses | Détecte un comportement anormal du modèle | Immédiatement | Alerte si multiplié ou divisé par plus de 2 par rapport à la référence (0,198 %) |
| Taux d'entrées rejetées par la validation | Détecte un changement de format ou une panne en amont | Immédiatement | Seuil à définir avec la banque |
| Disponibilité et latence de l'API (P95, taux d'erreurs) | Qualité de service | Immédiatement | P95 < 50 ms (mesuré à 2,4 ms avec validation en test local) |
| Rappel, précision, taux de faux positifs et de faux négatifs réels | Mesure la qualité réelle des décisions | Avec retard, lorsque les étiquettes arrivent | Mêmes tolérances que la non-régression : dégradation du rappel supérieure à 0,03, de la précision supérieure à 0,05, du FPR supérieure à 0,0003 |

## 3. Dérive des données et dérive de concept

| | Dérive des données (data drift) | Dérive de concept (concept drift) |
|---|---|---|
| Ce qui change | La distribution des entrées | La relation entre les entrées et la fraude (nouvelles techniques de fraude) |
| Détection immédiate | Oui : PSI, plage, taux signalé | Non |
| Détection | Surveillance des entrées | Uniquement quand les étiquettes arrivent : baisse du rappel ou de la PR-AUC |

## 4. Étiquettes retardées ou indisponibles

Les confirmations de fraude arrivent généralement après une réclamation du client, parfois des semaines plus tard (délai à confirmer avec la banque).

- **Suivi immédiat** : PSI, part hors plage, taux de transactions signalées, taux de rejet, latence, erreurs. Ils servent d'alerte précoce.
- **Suivi différé** : rappel, précision, taux de faux positifs et de faux négatifs réels, par cohortes de transactions une fois leurs étiquettes connues.
- Les indicateurs immédiats signalent un risque, pas une dégradation avérée.

## 5. Simulation des alertes

Script : `monitoring_report.py`. Référence calculée sur le jeu d'entraînement ; les lots surveillés sont des copies perturbées du jeu de validation. Référence : `reports/reference_monitoring.json`.

| Scénario | Variables en alerte | Variables en avertissement | Transactions hors plage | Taux signalé | Alertes |
|---|---|---|---|---|---|
| Référence (validation) | 0 | 0 | 0,009 % | 0,198 % | 0 |
| Bruit de 1 % sur V1-V28 | 0 | 0 | 0,009 % | 0,198 % | 0 |
| Bruit de 10 % sur V1-V28 | 0 | 0 | 0,586 % | 0,198 % | 0 |
| Montant +10 % | 0 | 0 | 0,009 % | 0,198 % | 0 |
| Montant extrême (1 000 000) | 1 | 0 | 100 % | 100 % | 3 |
| Décalage de +0,5 écart-type sur V1-V28 | 27 | 1 | 21,7 % | 0,186 % | 29 |

**Lecture.**
- **Aucune fausse alerte** sur les scénarios inoffensifs.
- **Montant extrême (anomalie A01)** : trois alertes se déclenchent (PSI de 8,301 sur `Amount`, 100 % des transactions hors plage, taux signalé à 100 % contre 0,20 %). La détection est en place, mais elle n'empêche pas le modèle de tout signaler : un garde-fou de décision (par exemple, envoyer ces cas en examen manuel) reste un choix métier à faire.
- **Décalage de distribution (anomalie A02)** : 29 alertes, avec un PSI atteignant 1,097 sur V1. Le taux de transactions signalées n'a presque pas varié (0,186 % contre 0,198 %), alors que le rappel avait baissé de 0,806 à 0,776 (rapport de robustesse). **Surveiller uniquement le taux de transactions signalées n'aurait rien détecté** : la surveillance des entrées est indispensable.

## 6. Réponse aux alertes

| Niveau | Action |
|---|---|
| Avertissement (PSI entre 0,10 et 0,25) | Observer, comparer avec les lots précédents, pas d'intervention |
| Alerte sur les entrées | Vérifier d'abord la chaîne de données en amont (changement de format, panne, nouveau canal) ; si la dérive est confirmée, constituer un jeu récent étiqueté |
| Alerte sur le taux signalé | Examiner un échantillon de transactions signalées, vérifier qu'il ne s'agit pas d'une attaque ou d'une panne |
| Rappel ou PR-AUC dégradé (étiquettes arrivées) | Réentraîner un candidat, le valider avec la procédure de non-régression (baseline, tolérances, décision), puis le déployer ; sinon revenir à la version précédente |

## 7. Outils

Pour ce prototype : des scripts et des rapports de métriques, avec des alertes simulées. Pour une exploitation réelle : Prometheus et Grafana sont adaptés (suivi de la latence, des erreurs, du taux de transactions signalées et des PSI, avec règles d'alerte). Ils ne sont pas implémentés ici.

## 8. Limites

- Les alertes sont simulées sur des perturbations artificielles du jeu de validation, non sur du trafic réel.
- Les seuils du PSI (0,10 et 0,25) sont des repères conventionnels ; le PSI dépend du découpage en classes et de la taille des lots. Les seuils de 1 % (hors plage) et de facteur 2 (taux signalé) sont des choix non validés : avec 10 % de bruit, 0,586 % des transactions sortent de la plage, sous le seuil sans le déclencher.
- La variable `Time` n'est pas surveillée : c'est un compteur relatif au début du jeu, qui sort toujours de la plage d'entraînement en production. Or le modèle l'utilise (anomalie A08).
- Un seul scénario de référence a été testé : le taux de fausses alertes réel en production est inconnu.
- Aucune surveillance du rappel réel n'est possible sans étiquettes : le délai de leur arrivée est à confirmer.
