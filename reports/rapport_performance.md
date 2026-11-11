# Rapport de performance et de sécurité opérationnelle

## 1. Protocole de mesure

| Élément | Valeur |
|---|---|
| Matériel | Ordinateur portable, 8 cœurs logiques, 8,2 Go de RAM, Linux sous WSL2 |
| Logiciels | Python 3.14.4, scikit-learn 1.9.1 |
| Modèle | Pipeline StandardScaler + régression logistique, rechargé depuis le disque |
| Données de mesure | Transactions du jeu de validation (le jeu de test n'est pas utilisé) |
| Prédiction isolée | 500 répétitions sur une même transaction, 1 appel d'échauffement exclu |
| Lots | Tailles 1, 100, 1 000 et 10 000, avec respectivement 500, 200, 50 et 20 répétitions |
| Temps | `time.perf_counter` ; P95 et P99 calculés avec `numpy.percentile` |
| Débit | Taille du lot divisée par la durée médiane d'un appel |
| Mémoire | `psutil` (RSS), dans un processus Python minimal : après imports, puis après chargement du modèle et une prédiction |
| Répétition | Script exécuté 3 fois, toutes les exécutions sont rapportées |

Deux latences sont distinguées. La **latence brute** mesure le modèle seul (`predict_proba`). La **latence avec validation** inclut en plus le contrôle du schéma et des types (`predict_from_dataframe`). Aucune des deux n'inclut le traitement HTTP, la sérialisation ni le réseau d'une vraie requête API : aucune API n'est déployée dans ce travail.

## 2. Résultats

### Latence d'une prédiction isolée (ms)

| Indicateur | Exéc. 1 | Exéc. 2 | Exéc. 3 | Seuil |
|---|---|---|---|---|
| Brute, médiane | 0,87 | 0,87 | 0,87 | |
| Brute, P95 | 1,10 | 0,94 | 1,28 | < 50 |
| Brute, P99 | 1,38 | 1,03 | 3,80 | |
| Avec validation, médiane | 1,97 | 1,91 | 1,91 | |
| Avec validation, P95 | 2,41 | 2,40 | 2,62 | < 50 |
| Avec validation, P99 | 2,59 | 2,55 | 3,83 | |

### Débit (prédictions par seconde, latence brute)

| Taille du lot | Exéc. 1 | Exéc. 2 | Exéc. 3 | Seuil |
|---|---|---|---|---|
| 1 | 1 145 | 1 154 | 1 085 | |
| 100 | 109 661 | 112 518 | 113 090 | >= 1 000 |
| 1 000 | 926 354 | 1 043 460 | 903 302 | |
| 10 000 | 5 761 198 | 5 999 338 | 5 743 001 | |

### Ressources

| Indicateur | Résultat | Seuil |
|---|---|---|
| Taille du modèle sauvegardé | 2 721 octets (0,0027 Mo) | < 10 Mo |
| Mémoire après imports (processus minimal) | 166,3 Mo | |
| Mémoire après chargement et prédiction | 168,6 Mo (surcoût du modèle : 2,3 Mo) | < 200 Mo |

## 3. Seuils d'acceptation et justification

Les seuils sont des hypothèses de travail fixées avant de mesurer et doivent être confirmés par la banque.

| Seuil | Justification | Résultat |
|---|---|---|
| Latence P95 < 50 ms | Une autorisation de paiement est une décision en temps réel : le modèle ne doit consommer qu'une petite part du temps total de traitement | Respecté : pire P95 de 1,28 ms (brute) et 2,62 ms (avec validation) |
| Débit >= 1 000 prédictions/s en lots de 100 | Marge sur un pic de trafic supposé | Respecté : environ 110 000 par seconde |
| Taille < 10 Mo | Déploiement léger, chargement rapide | Respecté |
| Mémoire < 200 Mo | Conteneur de petite taille | Respecté : 168,6 Mo, marge d'environ 15 % |

## 4. Interprétation

- **Latence.** Le pire P95 observé représente environ 5 % du budget de 50 ms. La validation des entrées coûte autant que le modèle (environ 1 ms de plus) mais le budget reste très largement respecté.
- **Variabilité.** Entre les trois exécutions, la médiane est stable et le débit unitaire varie d'environ 6 %. Le pic de P99 de la 3e exécution (3,8 ms) est un événement isolé, probablement lié à l'ordonnancement de WSL2 (non vérifié), sans effet sur le P95.
- **Débit.** Le coût fixe de chaque appel domine : passer de lots de 1 à lots de 100 multiplie le débit par environ 100. Si les transactions arrivent une à une, un seul processus traite environ 520 requêtes par seconde avec validation (valeur déduite de la latence médiane de 1,9 ms, non mesurée directement).
- **Mémoire.** Presque tout le coût vient des bibliothèques, pas du modèle.

## 5. Sécurité opérationnelle

| Contrôle | Test | Résultat |
|---|---|---|
| Champs supplémentaires | `test_extra_columns_ignored` | Colonne ignorée, prédictions identiques |
| Champs manquants, types incorrects, valeurs invalides | `test_invalid_input_rejected` (6 scénarios) | Rejet avec erreur explicite |
| Fichiers reçus invalides (vide, octets invalides, HTML, mauvaises colonnes) | `test_invalid_files_rejected` (4 scénarios) | Rejet avec `ValueError` explicite |
| Absence de données sensibles dans les journaux et messages d'erreur | `test_no_sensitive_values_in_errors_or_logs` | Aucune valeur de transaction dans les logs ni dans les erreurs |
| Absence de secrets dans le dépôt Git | `test_no_secrets_in_repository` | Aucun secret détecté (clés AWS, clés privées, jetons GitHub, secrets en clair) |
| Absence de données et de modèles dans le dépôt | `test_no_data_or_model_files_tracked` | Un fichier fautif détecté puis corrigé (A04) |

Le contrôle de secrets repose sur des motifs courants : il ne garantit pas l'absence de tout secret.

## 6. Anomalies et observations

| ID | Description | Gravité | Statut | Action |
|---|---|---|---|---|
| A04 | Fichier de métadonnées Windows (`Zone.Identifier`) versionné par erreur dans `data/` | Faible | Corrigé | Détecté par le test, retiré du suivi, motif ajouté au `.gitignore` |
| A05 | La mesure de mémoire du script `performance_report.py` donnait un surcoût de 0,0 Mo, mesure invalide (la mémoire était déjà occupée par le jeu de données) | Faible | Corrigé | Remplacée par `measure_memory.py`, qui mesure un processus minimal ; le verdict mémoire du script n'est pas retenu |
| A06 | Capacité unitaire avec validation estimée à environ 520 requêtes/s par processus, inférieure à l'hypothèse de 1 000 transactions/s | Moyenne | Ouvert | Lancer plusieurs processus ou regrouper les requêtes ; mesurer par test de charge si une API est déployée |

## 7. Limites

- Une seule machine, sous WSL2, avec un pic de variabilité possible lié à la virtualisation : les mesures ne se transposent pas telles quelles à un serveur de production.
- La même transaction est rejouée 500 fois : les effets de cache favorisent la latence mesurée.
- Aucune API n'est déployée : la latence HTTP, la sérialisation et la concurrence ne sont pas mesurées. Un test de charge (Locust ou JMeter) serait nécessaire.
- Le modèle est très léger (régression logistique à une trentaine de coefficients) : ces résultats ne s'appliquent pas à un modèle plus complexe.
- La capacité de 520 requêtes/s est une estimation déduite d'une médiane, pas une mesure directe.
- Les seuils sont des hypothèses non validées par la banque.

## 8. Conclusion

Tous les seuils d'acceptation sont respectés avec une large marge sur la latence, la taille et la mémoire. Le point de vigilance est la capacité de traitement unitaire avec validation (A06), à traiter avant tout déploiement à fort trafic. Les contrôles de sécurité opérationnelle passent, après correction de l'anomalie A04.
