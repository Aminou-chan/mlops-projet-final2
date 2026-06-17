"""
Task script pour générer les rapports d'explicabilité.

Ce script importe la classe `ReportBuilder` du package `src.explainability.report`
et construit le rapport d'explication global.

Usage:
    python tasks/report.py

Effets:
    - Exécute le workflow d'explicabilité défini dans `ReportBuilder`
    - Génère les fichiers de rapport associés
"""

from src.explainability.report import ReportBuilder

# Construire et générer le rapport d'explicabilité
ReportBuilder().build()
