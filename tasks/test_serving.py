"""
Task script pour tester le serving MLflow.

Ce script importe la fonction `main` du package `src.serving.test_serving`
et envoie des échantillons au serveur MLflow pour vérifier les prédictions.

Prérequis:
    Lancer au préalable le serveur dans un autre terminal :
        mlflow models serve -m "models:/california_housing_best_model@champion" \
            -p 5001 --env-manager local

Usage:
    python tasks/test_serving.py

Effets:
    - Envoie des échantillons au endpoint /invocations
    - Affiche les prédictions renvoyées par le modèle servi
"""

from src.serving.test_serving import main

# Test du serving MLflow
main()