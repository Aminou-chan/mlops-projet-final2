"""
Task script pour générer les explications locales SHAP.

Ce script importe `LocalFeatureAnalyzer` du package
`src.explainability.local_analysis` et exécute les analyses suivantes :
- Diagramme beeswarm global
- Waterfall plots pour les premiers échantillons

Usage:
    python tasks/local_analysis.py

Effets:
    - Charge le meilleur modèle entraîné
    - Calcule les valeurs SHAP sur un échantillon de test
    - Génére des visualisations dans `reports/figures/`
"""

from src.explainability.local_analysis import LocalFeatureAnalyzer

# Créer l'analyseur local SHAP
analyzer = LocalFeatureAnalyzer()

# Générer une vue globale de l'impact des features
analyzer.plot_beeswarm()

# Générer des explications locales pour les 3 premiers exemples
analyzer.explain_samples(3)
