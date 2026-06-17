# ==================== SCRIPT DE TÉLÉCHARGEMENT DES DONNÉES ====================
"""
Script autonome pour télécharger le dataset California Housing depuis scikit-learn
et le sauvegarder en CSV local.

Processus:
1. Appelle la fonction load_california_housing() pour télécharger les données
2. Sauvegarde le DataFrame en CSV dans le dossier data/raw/

Ce script peut être exécuté directement avec : python save_raw_data.py
"""

from load_data import load_california_housing

# ==================== TÉLÉCHARGEMENT ET SAUVEGARDE ====================

# Télécharger le dataset depuis scikit-learn
# (mise en cache local après le premier téléchargement)
df = load_california_housing()

# Créer le répertoire data/raw/ s'il n'existe pas et sauvegarder en CSV
df.to_csv("data/raw/california_housing.csv", index=False)

# Afficher un message de confirmation
print("Data saved in data/raw/california_housing.csv")
