# Rapport de non-régression

- Version de référence : dcc1ea7669a1efe35beec5291db1b43082312423
- Version évaluée : c48e469851af344ae561a9a44a228c875670dba9
- Modèle évalué : tree
- Seuil : 0.9582 (baseline : 0.9991)

| Métrique | Baseline | Version évaluée | Dégradation | Tolérance | Verdict |
|---|---|---|---|---|---|
| recall | 0.8061 | 0.8061 | -0.0000 | 0.03 | OK |
| precision | 0.6991 | 0.3015 | +0.3976 | 0.05 | hors tolérance |
| pr_auc | 0.6857 | 0.4891 | +0.1966 | 0.02 | hors tolérance |
| fpr | 0.0006 | 0.0032 | +0.0026 | 0.0003 | hors tolérance |

**Décision : REJETER**

- precision dégradé de 0.3976 (tolérance 0.05)
- pr_auc dégradé de 0.1966 (tolérance 0.02)
- fpr dégradé de 0.0026 (tolérance 0.0003)
