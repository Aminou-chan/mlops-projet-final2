# ==================== IMPORTS ====================
import os

import mlflow
from mlflow import MlflowClient


from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# ==================== CONFIGURATION ====================
# Nom du modèle dans le MLflow Model Registry
REGISTERED_MODEL_NAME = os.getenv(
    "REGISTERED_MODEL_NAME", "california_housing_best_model"
)

# Alias du modèle champion (utilisé pour le serving)
CHAMPION_ALIAS = os.getenv("MODEL_ALIAS", "champion")

# URI du tracking MLflow
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")


# ==================== FONCTION DE REGISTRATION ====================


def register_best_model(run_id: str) -> str:
    """
    Enregistre le meilleur modèle dans le MLflow Model Registry.

    Processus:
    1. Récupère le modèle depuis le run MLflow spécifié
    2. L'enregistre dans le Model Registry
    3. Lui attribue l'alias "champion" (référence pour le serving)

    Le Model Registry permet de:
    - Versionner les modèles
    - Les promouvoir dans différents stages (staging, production, etc.)
    - Les utiliser facilement en serving (via alias)
    - Tracker le cycle de vie complet du modèle

    Args:
        run_id (str): ID du run MLflow contenant le meilleur modèle

    Returns:
        str: Version du modèle enregistré dans le Model Registry

    Example:
        >>> version = register_best_model("run_12345")
        >>> print(version)
        "1"
    """
    # Construire l'URI du modèle dans ce run spécifique
    model_uri = f"runs:/{run_id}/model"

    # Configurer MLflow
    mlflow.set_tracking_uri(TRACKING_URI)

    # Enregistrer le modèle dans le Model Registry
    # (crée une nouvelle version du modèle)
    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME,
    )

    # Créer un client MLflow pour manipuler le Model Registry
    client = MlflowClient()

    # Attribuer l'alias "champion" à cette version
    # L'alias permet de référencer la meilleure version du modèle
    # sans connaître son numéro de version exact
    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias=CHAMPION_ALIAS,
        version=model_version.version,
    )

    # Afficher un message de confirmation
    print(
        f"Model registered: {REGISTERED_MODEL_NAME} "
        f"(version {model_version.version}, alias '{CHAMPION_ALIAS}')"
    )

    return model_version.version
