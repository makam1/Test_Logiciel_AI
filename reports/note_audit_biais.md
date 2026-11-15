# Note d'audit des biais et limites de l'évaluation

## 1. Objet

Identifier d'éventuelles disparités de performance du modèle entre groupes, lorsque des données fiables et légalement utilisables le permettent, et énoncer clairement ce qui ne peut pas être conclu.

## 2. Limite essentielle : pas d'analyse d'équité possible

Le jeu de données Credit Card Fraud contient `Time`, `Amount`, `Class` et 28 composantes anonymisées (V1 à V28, issues d'une analyse en composantes principales). Il ne contient **aucune caractéristique de personne** : ni âge, ni sexe, ni localisation, ni nationalité.

Il est donc **impossible de conclure, à partir de ce jeu de données, à l'équité ou à l'iniquité du système envers des groupes de personnes**. Aucun attribut démographique n'a été déduit ni inventé à partir des variables anonymisées : ce serait une spéculation sans fondement.

## 3. Ce qui a pu être analysé : disparités techniques par tranche de montant

Le montant d'une transaction n'est **pas** une caractéristique de personne. L'analyse suivante est une vérification de disparités techniques de performance, sans lien démontré avec l'équité.

**Protocole.** Jeu de validation (56 961 transactions, 98 fraudes), régression logistique, seuil de décision 0,9991 (choisi sur la validation). Quatre tranches de montant définies par les quartiles du jeu d'entraînement : moins de 5,74 (T1), de 5,74 à 22,05 (T2), de 22,05 à 77,50 (T3), plus de 77,50 (T4). Intervalles de confiance à 95 % de Wilson. Le jeu de test n'est pas utilisé. Script : `bias_audit.py`.

| Tranche | Transactions | Fraudes | Rappel [IC 95 %] | Taux de faux positifs [IC 95 %] | Précision [IC 95 %] | TP | FN | FP |
|---|---|---|---|---|---|---|---|---|
| T1 (plus faibles) | 14 318 | 43 | 0,8605 [0,7274 ; 0,9344] | 0,0013 [0,0008 ; 0,0020] | 0,6727 [0,5410 ; 0,7819] | 37 | 6 | 18 |
| T2 | 14 265 | 12 | 0,9167 [0,6461 ; 0,9851] | 0,0001 [0,0000 ; 0,0004] | 0,9167 [0,6461 ; 0,9851] | 11 | 1 | 1 |
| T3 | 14 210 | 5 | 0,8000 [0,3755 ; 0,9638] | 0,0001 [0,0000 ; 0,0004] | 0,8000 [0,3755 ; 0,9638] | 4 | 1 | 1 |
| T4 (plus élevés) | 14 168 | 38 | 0,7105 [0,5524 ; 0,8300] | 0,0010 [0,0006 ; 0,0017] | 0,6585 [0,5055 ; 0,7844] | 27 | 11 | 14 |

Les totaux (79 TP, 19 FN, 34 FP) correspondent aux résultats globaux de la validation.

## 4. Lecture des résultats

- **Rappel plus faible pour les montants élevés.** T4 détecte 71 % des fraudes contre 86 % pour T1, et passe sous le critère global de 0,80. Les intervalles de confiance se chevauchent : l'écart n'est pas établi statistiquement à cet effectif. Il concentre néanmoins 11 des 19 fraudes manquées (58 %).
- **Fausses alertes concentrées aux extrêmes.** 32 des 34 faux positifs (94 %) se trouvent en T1 et T4. Les intervalles de T1 et T4 ne chevauchent pas ceux de T2 et T3 : l'écart est plus solide. En valeur absolue, le taux reste très faible (environ 0,1 %), mais ces tranches sont celles où des clients légitimes sont le plus souvent bloqués.
- **T2 et T3 : aucune conclusion.** Avec 12 et 5 fraudes, les intervalles sont trop larges.
- **Le rappel mesure des transactions, pas des montants.** Les fraudes manquées dans la tranche la plus élevée sont peut-être les plus coûteuses : cela n'a pas été mesuré. Un rappel pondéré par le montant serait nécessaire pour le savoir.

## 5. Limites de cette analyse

- Seulement 98 fraudes en validation : les sous-groupes sont petits et les intervalles larges.
- Plusieurs indicateurs sont comparés sur quatre tranches : certains écarts peuvent apparaître par hasard (comparaisons multiples).
- Un seul seuil de décision est appliqué à toutes les tranches.
- Le jeu de données date de 2013 et provient de transactions européennes.
- Le chevauchement des intervalles de confiance est un repère prudent, non un test statistique formel.

## 6. Données complémentaires nécessaires pour un vrai audit d'équité

À obtenir de la banque, lorsque leur usage est légalement autorisé (la collecte et l'utilisation d'attributs sensibles sont encadrées, notamment par le RGPD) :

| Donnée | Intérêt |
|---|---|
| Pays ou région de résidence du titulaire | Disparités géographiques ; attention, la localisation peut servir de substitut à d'autres caractéristiques protégées |
| Ancienneté du client | Disparités entre nouveaux et anciens clients |
| Catégorie du commerçant, canal (en ligne, en magasin) | Disparités selon le contexte d'achat |
| Type de carte, devise | Disparités selon le produit |
| Tranche d'âge, uniquement si autorisée | Disparités entre classes d'âge |

**Indicateurs à calculer par groupe** : rappel, taux de faux positifs, précision, et écarts entre groupes avec leurs intervalles de confiance. Outils envisageables une fois les données disponibles : Fairlearn ou AI Fairness 360.

## 7. Recommandations

1. Ne pas communiquer de conclusion d'équité à partir de ce jeu de données.
2. Mesurer le rappel pondéré par le montant et examiner les fraudes manquées dans la tranche des montants élevés (anomalie A09).
3. Obtenir un jeu de données avec attributs de groupe, et refaire l'audit avant un déploiement réel.
4. Suivre en production le taux de faux positifs par segment, car ce sont les clients légitimes qui en subissent les conséquences.
