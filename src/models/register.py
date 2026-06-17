import mlflow
from mlflow import MlflowClient


REGISTERED_MODEL_NAME = "california_housing_best_model"
CHAMPION_ALIAS = "champion"


def register_best_model(run_id: str) -> str:
    """
    Enregistre le meilleur modele dans le MLflow Model Registry
    et lui pose l'alias 'champion' (reference utilisee ensuite pour le serving).
    """
    model_uri = f"runs:/{run_id}/model"

    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME,
    )

    client = MlflowClient()
    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias=CHAMPION_ALIAS,
        version=model_version.version,
    )

    print(
        f"Model registered: {REGISTERED_MODEL_NAME} "
        f"(version {model_version.version}, alias '{CHAMPION_ALIAS}')"
    )
    return model_version.version