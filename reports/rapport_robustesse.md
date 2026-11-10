# Rapport de robustesse et de sensibilité

## 1. Protocole

- Modèle : régression logistique (pipeline StandardScaler + modèle, `class_weight="balanced"`), seuil de décision 0,9991 choisi sur la validation.
- Les perturbations sont appliquées au jeu de **validation** (56 961 transactions, 98 fraudes). Le jeu de test n'est pas utilisé.
- Graine aléatoire : 123. Résultats reproductibles avec `python robustness_report.py`.
- Amplitudes documentées :
  - bruit gaussien d'écart-type 1 % puis 10 % de l'écart-type de chaque colonne, sur V1 à V28 uniquement (Time et Amount ne reçoivent pas ce bruit, car leur signification diffère de celle des composantes anonymisées) ;
  - Amount multiplié par 1,10 ; Amount fixé à 1 000 000 pour toutes les transactions ;
  - décalage de V1 à V28 de +0,5 écart-type ;
  - 1 % de cellules rendues manquantes au hasard.
- Indicateurs : variation moyenne de la probabilité prédite, pourcentage de transactions changeant de classe, rappel, précision, FPR, nombre de lignes devenues invalides.

## 2. Comportement attendu par scénario

| Scénario | Comportement attendu | Test pytest | Résultat |
|---|---|---|---|
| Valeurs manquantes (NaN) | Rejeter (ValueError) | test_invalid_input_rejected[valeurs_manquantes] | Conforme |
| Valeurs infinies | Rejeter | test_invalid_input_rejected[valeurs_infinies] | Conforme |
| Type non numérique | Rejeter | test_invalid_input_rejected[type_incorrect] | Conforme |
| Colonne absente | Rejeter | test_invalid_input_rejected[colonne_absente] | Conforme |
| Colonne dupliquée | Rejeter | test_invalid_input_rejected[colonne_dupliquee] | Conforme |
| Données vides | Rejeter | test_invalid_input_rejected[donnees_vides] | Conforme |
| Colonne supplémentaire | Traiter (colonne ignorée) | test_extra_columns_ignored | Conforme |
| Montant extrême | Traiter avec alerte | test_probability_valid_for_any_amount | Sortie valide, mais comportement du modèle non satisfaisant (A01) |
| Perturbation limitée | Traiter et mesurer la sensibilité | test_small_perturbation_sensitivity | Conforme (tolérances : variation moyenne < 0,02 ; changement de classe < 5 %) |
| Décalage de distribution | Traiter avec alerte | test_distribution_shift_is_processed_and_measurable | Traité ; alerte non encore implémentée (A02) |

## 3. Résultats sur la validation

| Scénario | Variation moyenne de la probabilité | Changement de classe | Rappel | Précision | FPR | FP | FN | Lignes invalides |
|---|---|---|---|---|---|---|---|---|
| Référence | 0 | 0 % | 0,8061 | 0,6991 | 0,0006 | 34 | 19 | 0 |
| Bruit 1 % sur V1-V28 | 0,0016 | 0 % | 0,8061 | 0,6991 | 0,0006 | 34 | 19 | 0 |
| Bruit 10 % sur V1-V28 | 0,0165 | 0 % | 0,8061 | 0,6991 | 0,0006 | 34 | 19 | 0 |
| Montant +10 % | 0,0028 | 0 % | 0,8061 | 0,6991 | 0,0006 | 34 | 19 | 0 |
| Montant extrême (1 000 000) | 0,9326 | 99,80 % | 1,0000 | 0,0017 | 1,0000 | 56 863 | 0 | 0 |
| Décalage +0,5 écart-type sur V1-V28 | 0,0551 | 1,23 % | 0,7755 | 0,7170 | 0,0005 | 30 | 22 | 0 |
| 1 % de valeurs manquantes | non applicable | non applicable | non applicable | non applicable | non applicable | non applicable | non applicable | 14 989 (environ 26 %) |

## 4. Interprétation

**Robustesse au bruit.** Même avec un bruit de 10 % de l'écart-type, aucune transaction ne change de classe. Le seuil très élevé (0,9991) isole un petit groupe de transactions très suspectes, peu sensibles à de petites perturbations.

**Sensibilité aux valeurs hors plage.** Avec un montant de 1 000 000, presque toutes les transactions sont déclarées frauduleuses : le rappel monte à 1,0 uniquement parce que tout est signalé, et la précision tombe à 0,17 %. Le modèle linéaire extrapole sans garde-fou. Dans un cas réel, une transaction isolée de montant aberrant serait signalée, ce qui peut être acceptable métier, mais il s'agit d'un artefact d'extrapolation et non d'un comportement appris.

**Sensibilité à la dérive.** Un décalage de +0,5 écart-type fait perdre 3 fraudes détectées sur 98 et fait passer le rappel sous le seuil d'acceptation de 0,80. Le critère d'acceptation n'est donc pas robuste à la dérive des données.

**Entrées incomplètes.** La politique de rejet est stricte : une seule valeur manquante rend la transaction inutilisable. Avec 1 % de cellules manquantes, environ 26 % des transactions seraient rejetées.

## 5. Anomalies

| ID | Description | Gravité | Statut | Action |
|---|---|---|---|---|
| A01 | Montant hors plage : toutes les transactions sont signalées comme fraude, sans alerte | Élevée | Ouvert | Ajouter un garde-fou de plausibilité fondé sur la plage observée à l'entraînement, avec alerte (Activité 10) |
| A02 | Un décalage de distribution de +0,5 écart-type fait passer le rappel sous 0,80 | Moyenne | Ouvert | Surveiller la dérive des variables et déclencher une alerte (Activité 10) |
| A03 | 1 % de valeurs manquantes rend environ 26 % des transactions inutilisables | Moyenne | Ouvert | Décision à prendre : garantir la complétude en amont, ou imputer selon une règle documentée avec alerte |

## 6. Limites

- Une seule graine aléatoire : un résultat plus solide nécessiterait de répéter les perturbations avec plusieurs graines.
- Seulement 98 fraudes dans la validation : les variations de rappel de quelques points correspondent à quelques transactions.
- Les perturbations sont synthétiques et leurs amplitudes arbitraires. Les variables V1 à V28 étant anonymisées, il est impossible de dire quelles perturbations sont réalistes en pratique.
- Le test Hypothesis vérifie le contrat de sortie (probabilité valide) et non la pertinence de la prédiction.
- Le montant extrême est appliqué à toutes les transactions simultanément, scénario volontairement pessimiste.
