import types
from unittest.mock import MagicMock

import src.models.register as reg


def test_register_best_model(monkeypatch):
    """On mocke MLflow : aucun vrai registre n'est touche."""
    fake_version = types.SimpleNamespace(version="7")
    register_mock = MagicMock(return_value=fake_version)
    client_mock = MagicMock()

    monkeypatch.setattr(reg.mlflow, "set_tracking_uri", lambda *a, **k: None)
    monkeypatch.setattr(reg.mlflow, "register_model", register_mock)
    monkeypatch.setattr(reg, "MlflowClient", lambda: client_mock)

    version = reg.register_best_model("RUN123")

    assert version == "7"
    # l'URI du modele est bien construite a partir du run_id
    register_mock.assert_called_once_with(
        model_uri="runs:/RUN123/model",
        name=reg.REGISTERED_MODEL_NAME,
    )
    # l'alias champion est pose sur la bonne version
    client_mock.set_registered_model_alias.assert_called_once_with(
        name=reg.REGISTERED_MODEL_NAME,
        alias=reg.CHAMPION_ALIAS,
        version="7",
    )
