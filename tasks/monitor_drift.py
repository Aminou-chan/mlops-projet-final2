"""
Task script pour générer le rapport de dérive (data drift) avec Evidently.

Ce script importe la fonction `main` du package
`src.monitoring.simulate_production`. Il simule des données de production
(dérive réaliste) puis génère le rapport de dérive.

Usage:
    python tasks/monitor_drift.py

Effets:
    - Simule des données de production (inflation des revenus + décalage géo)
    - Compare ces données aux données d'entraînement (référence)
    - Génère un rapport HTML dans `reports/drift/data_drift_report.html`
"""

from src.monitoring.simulate_production import main

# Génération du rapport de dérive sur données de production simulées
main()