"""
Task script pour calculer et stocker l'importance globale des features.

Ce script importe la classe `GlobalFeatureAnalyzer` du package
`src.explainability.global_analysis` et exécute l'analyse globale.

Usage:
    python tasks/global_analysis.py

Effets:
    - Charge le meilleur modèle entraîné
    - Calcule les importances des features
    - Sauvegarde le résultat dans `reports/explainability/global_importance.csv`
    - Génère un graphique PNG dans `reports/figures/`
"""

from src.explainability.global_analysis import GlobalFeatureAnalyzer

# Exécuter l'analyse globale des importances de features
GlobalFeatureAnalyzer().compute_global_importance()
