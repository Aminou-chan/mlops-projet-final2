import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


# Nom de la colonne cible (variable à prédire)
TARGET = "MedHouseVal"

# Colonnes sur lesquelles appliquer une transformation logarithmique
# Ces colonnes ont des distributions asymétriques et bénéficient d'une log transformation
LOG_COLUMNS = ["AveRooms", "AveBedrms", "Population", "AveOccup"]


def preprocess_data(input_path: str = "data/raw/california_housing.csv"):
    """
    Prétraite le dataset California Housing et crée deux pipelines distincts.

    Cette fonction effectue les opérations suivantes :
    1. Charge les données brutes
    2. Sépare les features de la cible
    3. Divise en train/test (80/20)
    4. Crée deux pipelines parallèles:

       Pipeline Linéaire (pour modèles linéaires):
       - Application d'une transformation logarithmique sur 4 colonnes
       - Normalisation robuste (RobustScaler) de toutes les features
       - Sauvegarde du scaler pour utilisation en prédiction

       Pipeline Tree (pour modèles tree-based):
       - Données brutes sans transformation (arbres insensibles à l'échelle)

    Exporte en fichiers CSV:
    - data/processed/X_train_linear.csv & X_test_linear.csv (features linéaires normalisées)
    - data/processed/X_train_tree.csv & X_test_tree.csv (features tree brutes)
    - data/processed/y_train.csv & y_test.csv (cibles)
    - models/robust_scaler.pkl (scaler sauvegardé)

    Args:
        input_path (str): Chemin du fichier CSV brut (défaut: data/raw/california_housing.csv)

    Returns:
        None (les données sont sauvegardées en fichiers)
    """

    # Charger le dataset brut
    df = pd.read_csv(input_path)

    # Séparer les features (X) de la cible (y)
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]

    # Diviser en ensemble d'entraînement (80%) et test (20%)
    # Le même split est utilisé pour les deux pipelines (reproductibilité)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Pipeline Linéaire
    # Ce pipeline est destiné aux modèles linéaires et autres algorithmes
    # sensibles à l'échelle des variables

    # Copier les données brutes
    X_train_linear = X_train.copy()
    X_test_linear = X_test.copy()

    # Appliquer une transformation logarithmique aux colonnes sélectionnées
    # log1p = log(1 + x) évite les problèmes avec log(0)
    for col in LOG_COLUMNS:
        X_train_linear[col] = np.log1p(X_train_linear[col])
        X_test_linear[col] = np.log1p(X_test_linear[col])

    # Normaliser les données avec RobustScaler (résistant aux outliers)
    # Utilise la médiane et l'IQR au lieu de la moyenne et écart-type
    scaler = RobustScaler()

    # Fit sur train, transform sur train et test
    X_train_linear = scaler.fit_transform(X_train_linear)
    X_test_linear = scaler.transform(X_test_linear)

    # Sauvegarder le scaler pour l'utiliser en prédiction future
    with open("models/robust_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Exporter les données linéaires normalisées en CSV
    pd.DataFrame(X_train_linear, columns=X_train.columns).to_csv(
        "data/processed/X_train_linear.csv", index=False
    )

    pd.DataFrame(X_test_linear, columns=X_test.columns).to_csv(
        "data/processed/X_test_linear.csv", index=False
    )

    # Pipeline Tree
    # Ce pipeline est destiné aux modèles tree-based (Decision Trees, Random Forests, etc.)
    # Les arbres sont insensibles à l'échelle, donc pas de normalisation nécessaire
    # Les log transformations ne sont pas bénéfiques pour les arbres

    # Copier les données brutes sans transformation
    X_train_tree = X_train.copy()
    X_test_tree = X_test.copy()

    # Exporter les données brutes en CSV (pas de preprocessing)
    X_train_tree.to_csv("data/processed/X_train_tree.csv", index=False)

    X_test_tree.to_csv("data/processed/X_test_tree.csv", index=False)

    # Sauvegarder les targets séparées pour train et test
    y_train.to_csv("data/processed/y_train.csv", index=False)

    y_test.to_csv("data/processed/y_test.csv", index=False)

    # Afficher un message de confirmation
    print("Preprocessing completed successfully.")
