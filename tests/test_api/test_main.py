from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)

VALID = {
    "MedInc": 8.3,
    "HouseAge": 41,
    "AveRooms": 6.9,
    "AveBedrms": 1.0,
    "Population": 322,
    "AveOccup": 2.5,
    "Latitude": 37.88,
    "Longitude": -122.23,
}


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert "message" in r.json()


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_ok():
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"med_house_val", "predicted_price_usd"}
    # coherence : le prix en $ = valeur brute * 100 000
    assert body["predicted_price_usd"] == body["med_house_val"] * 100_000


def test_predict_missing_field():
    bad = {k: v for k, v in VALID.items() if k != "MedInc"}
    assert client.post("/predict", json=bad).status_code == 422


def test_predict_wrong_type():
    bad = dict(VALID, MedInc="beaucoup")
    assert client.post("/predict", json=bad).status_code == 422
