from sklearn.datasets import fetch_california_housing
import pandas as pd


def load_california_housing() -> pd.DataFrame:
    """
    Charge le dataset California Housing depuis scikit-learn.

    Cette fonction télécharge le dataset California Housing depuis le dépôt scikit-learn
    et le retourne sous forme d'un DataFrame pandas. Ce dataset contient des données
    sur les prix immobiliers en Californie.

    Le dataset contient 8 features:
        - MedInc: Revenu médian du quartier
        - HouseAge: Âge médian des maisons
        - AveRooms: Nombre moyen de pièces par logement
        - AveBedrms: Nombre moyen de chambres par logement
        - Population: Population du quartier
        - AveOccup: Taux d'occupation moyen
        - Latitude: Latitude géographique
        - Longitude: Longitude géographique

    Et 1 cible:
        - MedHouseVal: Valeur médiane des maisons (en unités de 100k $)

    Returns:
        pd.DataFrame: DataFrame contenant le dataset complet avec features et cible

    Note:
        La première execution télécharge le dataset depuis internet (~3.5 MB).
        Les téléchargements suivants utilisent le cache local.
    """
    # Charger le dataset depuis scikit-learn en tant que DataFrame (as_frame=True)
    housing = fetch_california_housing(as_frame=True)

    # Extraire le DataFrame qui contient features et cible
    df = housing.frame

    # Retourner le DataFrame complet
    return df
