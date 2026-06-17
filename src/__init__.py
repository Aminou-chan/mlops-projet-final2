"""
Package source principal du projet ImmoPrix - MLOps pour prédiction de prix immobilier.

Structure du package :

    src/
    ├── api/              → API REST (FastAPI) et interface web (Streamlit)
    │   ├── main.py       → Endpoints REST pour prédictions
    │   └── streamlit_app.py → Interface utilisateur interactive
    │
    ├── data/             → Modules de chargement et préparation des données
    │   ├── load_data.py  → Chargement du dataset California Housing
    │   └── save_raw_data.py → Sauvegarde du dataset brut
    │
    ├── features/         → Prétraitement et normalisation
    │   └── preprocessing.py → Pipelines linéaire et tree
    │
    ├── models/           → Entraînement, évaluation et registry du modèle
    │   ├── train.py      → Orchestration de l'entraînement, tuning, comparaison
    │   ├── evaluate.py   → Métriques (RMSE, MAE, R²)
    │   └── register.py   → Enregistrement dans MLflow Model Registry
    │
    ├── explainability/   → Explainabilité des prédictions
    │   ├── global_analysis.py → Importances globales des features
    │   └── local_analysis.py  → Explications locales SHAP
    │
    ├── validation/       → Validation des données brutes
    │   └── data_validation.py → Great Expectations (GX)
    │
    ├── monitoring/       → Détection de dérives en production
    │   ├── drift_monitoring.py → Analyse de data drift (Evidently)
    │   └── simulate_production.py → Simulation de données de production
    │
    └── serving/          → Tests du modèle déployé
        └── test_serving.py → Tests du endpoint MLflow REST

Flux MLOps principal:
    1. data/save_raw_data.py → Télécharge et sauvegarde les données brutes
    2. validation/data_validation.py → Valide les données (Great Expectations)
    3. features/preprocessing.py → Prépare deux pipelines (linéaire + tree)
    4. models/train.py → Entraîne 3 modèles et trouve le meilleur
    5. models/register.py → Enregistre le meilleur modèle dans le Model Registry
    6. api/main.py → Expose le modèle via API REST
    7. api/streamlit_app.py → Interface web pour les prédictions
    8. monitoring/drift_monitoring.py → Détecte les dérives en production
    9. explainability/* → Explique les prédictions (global + local)

Dépendances clés:
    - scikit-learn → Modèles ML
    - MLflow → Tracking et serving
    - FastAPI → API REST
    - Streamlit → Interface web
    - Great Expectations → Validation de données
    - Evidently → Monitoring de dérives
    - SHAP → Explainabilité locale
"""
