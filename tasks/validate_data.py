"""
Task script pour valider les données brutes avec Great Expectations.

Ce script importe la fonction `main` du package
`src.validation.data_validation` et exécute la validation du dataset.

Usage:
    python tasks/validate_data.py

Effets:
    - Charge le dataset brut
    - Vérifie le contrat de données (colonnes, valeurs manquantes, plages)
    - Affiche un résumé et renvoie un code d'erreur si la validation échoue
"""

from src.validation.data_validation import main

# Exécution de la validation des données
main()