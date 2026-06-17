"""
Module de simulation de données de production pour tester la détection de dérive.

Contexte:
- Dans un vrai système, les données de production arrivent au fil du temps
- Elles peuvent s'éloigner des données d'entraînement (dérive) pour plusieurs raisons:
  * Inflation économique (revenus augmentent)
  * Changement démographique (population se déplace)
  * Évolution du marché (prix changent)
  * Biais de sélection (nouvelles clientèles)

Limitation de ce projet:
- On ne dispose pas de vraies données de production
- Solution : SIMULER une dérive réaliste sur les données existantes
- Permet de démontrer que les outils de monitoring (Evidently) fonctionnent

Dérives simulées:
1. Inflation des revenus : MedInc augmente de +20%
   → Simule une augmentation du coût de la vie et des revenus

2. Décalage géographique : Latitude/Longitude changent
   → Simule un marché immobilier qui s'étend à d'autres régions
"""

# ==================== IMPORTS ====================
import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.monitoring.drift_monitoring import drift_summary, generate_drift_report

# Charger les variables d'environnement
load_dotenv()

# ==================== CONFIGURATION ====================
# Chemin du dataset de référence (données d'entraînement)
REFERENCE_PATH = os.getenv("DRIFT_REFERENCE_PATH", "data/processed/X_train_tree.csv")

# Chemin du dataset de base à utiliser comme point de départ
BASE_PATH = os.getenv("DRIFT_CURRENT_PATH", "data/processed/X_test_tree.csv")

# Chemin de sortie du rapport de dérive
REPORT_PATH = os.getenv("DRIFT_REPORT_PATH", "reports/drift/data_drift_report.html")


# ==================== FONCTION DE SIMULATION ====================


def simulate_production_data(
    df: pd.DataFrame,
    income_inflation: float = 0.20,
    geo_shift: float = 1.5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Applique une dérive réaliste à un DataFrame pour imiter les données de production.

    Simule deux transformations de dérive :

    1. Inflation économique (MedInc)
       - Les revenus augmentent de `income_inflation` (ex: 0.20 = +20%)
       - Ajout d'un bruit gaussien pour plus de réalisme
       - Simule une économie en croissance, inflation, etc.

    2. Décalage géographique (Latitude/Longitude)
       - Les latitudes augmentent de `geo_shift`
       - Les longitudes diminuent de `geo_shift`
       - Simule un marché qui s'étend vers le nord et l'est
       - Utile pour tester la sensibilité aux variables de localisation

    Args:
        df (pd.DataFrame): DataFrame original (données de base)
        income_inflation (float): Hausse relative appliquée à MedInc (défaut: 0.20)
                                  Exemple : 0.20 = +20%
        geo_shift (float): Décalage appliqué à Latitude/Longitude (défaut: 1.5)
                          Positive pour Latitude, négative pour Longitude
        seed (int): Graine pour le générateur aléatoire (reproductibilité)

    Returns:
        pd.DataFrame: DataFrame transformé avec la dérive simulée

    Example:
        >>> df_prod = simulate_production_data(df_test, income_inflation=0.20, geo_shift=1.5)
        >>> print(df_prod[['MedInc', 'Latitude']].head())
        # Les valeurs seront augmentées selon les transformations
    """
    # Créer un générateur aléatoire avec seed pour reproductibilité
    rng = np.random.default_rng(seed)

    # Copier le DataFrame original (ne pas le modifier)
    simulated = df.copy()

    # ===== Transformation 1 : Inflation économique =====
    # Augmenter les revenus de income_inflation (ex: +20%)
    simulated["MedInc"] = simulated["MedInc"] * (1 + income_inflation)

    # Ajouter un bruit gaussien pour plus de réalisme
    # (pas juste une multiplication uniforme)
    simulated["MedInc"] += rng.normal(0, 0.1, len(simulated))

    # ===== Transformation 2 : Décalage géographique =====
    # Augmenter les latitudes (décalage vers le nord)
    simulated["Latitude"] = simulated["Latitude"] + geo_shift

    # Diminuer les longitudes (décalage vers l'est)
    simulated["Longitude"] = simulated["Longitude"] - geo_shift

    # Retourner le DataFrame transformé
    return simulated


# ==================== FONCTION PRINCIPALE ====================


def main() -> None:
    """
    Point d'entrée du module : simule la dérive et génère un rapport.

    Processus:
    1. Charger le dataset de référence (entraînement)
    2. Charger le dataset de base (test)
    3. Appliquer les transformations de dérive
    4. Générer le rapport de dérive Evidently
    5. Afficher un résumé des résultats
    """
    # Charger le dataset de référence (entraînement - baseline)
    reference = pd.read_csv(REFERENCE_PATH)

    # Charger le dataset de base (test - point de départ)
    base = pd.read_csv(BASE_PATH)

    # Appliquer les transformations de dérive
    # (inflation +20% + décalage géographique)
    production = simulate_production_data(base)

    # Générer le rapport en comparant référence vs production simulée
    my_eval = generate_drift_report(reference, production, REPORT_PATH)

    # Extraire le résumé de dérive
    summary = drift_summary(my_eval)

    # Afficher les résultats
    print("Donnees de production simulees (inflation +20 % + decalage geo).")
    print(f"Rapport de derive : {REPORT_PATH}")
    print(
        f"Colonnes ayant derive : {int(summary['count'])} "
        f"({summary['share'] * 100:.0f}% des colonnes)"
    )


if __name__ == "__main__":
    main()
