import os

import pandas as pd
import requests
import streamlit as st

from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")

# Configuration de la page Streamlit
st.set_page_config(
    page_title="ImmoPrix",
    page_icon="🏠",
    layout="wide",
)


def check_api():
    """
    Effectue un health check sur l'API.
    Permet de vérifier si l'API FastAPI est accessible et opérationnelle.

    Cette fonction contacte l'endpoint /health pour confirmer que:
    - L'API est en cours d'exécution
    - Le modèle est correctement chargé

    Returns:
        bool: True si l'API répond correctement (status 200), False sinon

    Note:
        Utilise un timeout de 3 secondes pour éviter les blocages prolongés.
    """
    try:
        # Effectuer une requête GET sur l'endpoint /health avec timeout
        r = requests.get(f"{API_URL}/health", timeout=3)
        # Retourner True si le statut est 200 (OK)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        # Si une erreur de connexion survient, retourner False
        return False


# ==================== INTERFACE - BARRE LATÉRALE ====================
# Section de saisie des features dans la barre latérale gauche
# Les utilisateurs ajustent ces valeurs pour configurer le district à analyser

st.sidebar.header("Caracteristiques du district")

# Inputs numériques pour les caractéristiques du district
# Les valeurs par défaut correspondent aux moyennes du dataset California Housing
med_inc = st.sidebar.slider("Revenu median (10k $)", 0.5, 15.0, 3.87, 0.1)
house_age = st.sidebar.slider("Age median des maisons", 1, 52, 28)
ave_rooms = st.sidebar.slider("Pieces / logement", 1.0, 15.0, 5.43, 0.1)
ave_bedrms = st.sidebar.slider("Chambres / logement", 0.5, 5.0, 1.10, 0.05)
ave_occup = st.sidebar.slider("Occupation moyenne", 0.5, 10.0, 3.07, 0.1)
population = st.sidebar.number_input("Population du secteur", value=1425, step=10)

# Section de localisation - latitude et longitude pour afficher la carte
st.sidebar.subheader("Localisation")
latitude = st.sidebar.slider("Latitude", 32.5, 42.0, 35.63, 0.01)
longitude = st.sidebar.slider("Longitude", -124.4, -114.3, -119.57, 0.01)

# ==================== INTERFACE - CORPS PRINCIPAL ====================
# Affichage du titre et de la description de l'application

st.title("🏠 ImmoPrix — Estimation de prix immobilier")
st.caption(
    "Prediction du prix median d'un logement en Californie a partir du modele MLOps."
)

# Vérifier l'état de connexion de l'API et afficher le statut approprié
if check_api():
    st.success(f"API connectee ({API_URL})", icon="✅")
else:
    st.warning(
        f"API injoignable sur {API_URL} — lance le conteneur avant de predire.",
        icon="⚠️",
    )

st.divider()  # Séparateur visuel

# ==================== INTERFACE - CONTENU PRINCIPAL ====================
# Disposition en deux colonnes : carte à gauche, résultats à droite

col_map, col_result = st.columns([1, 1])

# COLONNE 1 : Carte de visualisation
with col_map:
    st.subheader("Localisation du district")
    # Afficher une carte interactive avec les coordonnées sélectionnées
    st.map(pd.DataFrame({"lat": [latitude], "lon": [longitude]}), zoom=5)

# COLONNE 2 : Zone de prédiction et résultats
with col_result:
    st.subheader("Estimation")

    # Bouton principal pour déclencher la prédiction
    if st.button("Predire le prix", type="primary", use_container_width=True):
        # Construire le payload JSON avec toutes les features nécessaires
        # Cet ordre doit correspondre à ce qu'attend l'API
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
            # Effectuer la requête POST à l'API avec affichage d'un spinner de chargement
            with st.spinner("Calcul en cours..."):
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
                # Lever une exception si le statut HTTP indique une erreur
                response.raise_for_status()
        except requests.exceptions.RequestException as e:
            # En cas d'erreur de connexion ou de réponse
            st.error("Impossible de contacter l'API.")
            st.caption(str(e))
        else:
            # Traitement du résultat en cas de succès
            result = response.json()
            # Afficher la prédiction en dollars avec formatage (virgules, etc.)
            st.metric(
                label="Prix median estime",
                value=f"{result['predicted_price_usd']:,.0f} $",
            )
            # Afficher aussi la valeur brute du modèle (en unités de 100k$) à titre informatif
            st.caption(
                f"Valeur brute du modele (MedHouseVal) : {result['med_house_val']:.3f}"
            )
    else:
        # Message d'aide affiché avant que l'utilisateur ne clique sur le bouton
        st.info("Regle les caracteristiques a gauche puis clique sur **Predire**.")
