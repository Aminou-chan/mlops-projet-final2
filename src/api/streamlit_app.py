import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="ImmoPrix",
    page_icon="🏠",
    layout="wide",
)


def check_api():
    """Petit ping sur /health pour afficher l'etat de l'API."""
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False


# ---------- Barre laterale : saisie des features ----------
st.sidebar.header("Caracteristiques du district")

med_inc = st.sidebar.slider("Revenu median (10k $)", 0.5, 15.0, 3.87, 0.1)
house_age = st.sidebar.slider("Age median des maisons", 1, 52, 28)
ave_rooms = st.sidebar.slider("Pieces / logement", 1.0, 15.0, 5.43, 0.1)
ave_bedrms = st.sidebar.slider("Chambres / logement", 0.5, 5.0, 1.10, 0.05)
ave_occup = st.sidebar.slider("Occupation moyenne", 0.5, 10.0, 3.07, 0.1)
population = st.sidebar.number_input("Population du secteur", value=1425, step=10)

st.sidebar.subheader("Localisation")
latitude = st.sidebar.slider("Latitude", 32.5, 42.0, 35.63, 0.01)
longitude = st.sidebar.slider("Longitude", -124.4, -114.3, -119.57, 0.01)

# ---------- En-tete ----------
st.title("🏠 ImmoPrix — Estimation de prix immobilier")
st.caption("Prediction du prix median d'un logement en Californie a partir du modele MLOps.")

# Etat de l'API
if check_api():
    st.success(f"API connectee ({API_URL})", icon="✅")
else:
    st.warning(f"API injoignable sur {API_URL} — lance le conteneur avant de predire.", icon="⚠️")

st.divider()

# ---------- Corps : carte + resultat ----------
col_map, col_result = st.columns([1, 1])

with col_map:
    st.subheader("Localisation du district")
    st.map(pd.DataFrame({"lat": [latitude], "lon": [longitude]}), zoom=5)

with col_result:
    st.subheader("Estimation")

    if st.button("Predire le prix", type="primary", use_container_width=True):
        payload = {
            "MedInc": med_inc,
            "HouseAge": house_age,
            "AveRooms": ave_rooms,
            "AveBedrms": ave_bedrms,
            "Population": population,
            "AveOccup": ave_occup,
            "Latitude": latitude,
            "Longitude": longitude,
        }

        try:
            with st.spinner("Calcul en cours..."):
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
                response.raise_for_status()
        except requests.exceptions.RequestException as e:
            st.error("Impossible de contacter l'API.")
            st.caption(str(e))
        else:
            result = response.json()
            st.metric(
                label="Prix median estime",
                value=f"{result['predicted_price_usd']:,.0f} $",
            )
            st.caption(
                f"Valeur brute du modele (MedHouseVal) : {result['med_house_val']:.3f}"
            )
    else:
        st.info("Regle les caracteristiques a gauche puis clique sur **Predire**.")