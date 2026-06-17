"""Simulation de donnees de production pour tester la detection de derive.

Dans un vrai systeme, les donnees de production arrivent au fil du temps et
peuvent s'eloigner de celles d'entrainement (inflation, changement de zone
geographique, etc.). Comme on ne dispose pas de telles donnees dans ce projet,
on les SIMULE en appliquant une derive realiste a des donnees existantes,
afin de demontrer qu'Evidently la detecte bien.

Derive simulee :
  - inflation des revenus : MedInc augmente de +20 %
  - decalage geographique : Latitude / Longitude deplacees (autre zone)
"""

import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.monitoring.drift_monitoring import drift_summary, generate_drift_report

load_dotenv()

REFERENCE_PATH = os.getenv("DRIFT_REFERENCE_PATH", "data/processed/X_train_tree.csv")
BASE_PATH = os.getenv("DRIFT_CURRENT_PATH", "data/processed/X_test_tree.csv")
REPORT_PATH = os.getenv("DRIFT_REPORT_PATH", "reports/drift/data_drift_report.html")


def simulate_production_data(
    df: pd.DataFrame,
    income_inflation: float = 0.20,
    geo_shift: float = 1.5,
    seed: int = 42,
) -> pd.DataFrame:
    """Applique une derive realiste a un DataFrame pour imiter la production.

    - income_inflation : hausse relative appliquee a MedInc (0.20 = +20 %)
    - geo_shift : decalage applique a la latitude/longitude (autre zone)
    """
    rng = np.random.default_rng(seed)
    simulated = df.copy()

    # 1) Inflation des revenus (+ un peu de bruit)
    simulated["MedInc"] = simulated["MedInc"] * (1 + income_inflation)
    simulated["MedInc"] += rng.normal(0, 0.1, len(simulated))

    # 2) Decalage geographique : on simule des biens d'une autre region
    simulated["Latitude"] = simulated["Latitude"] + geo_shift
    simulated["Longitude"] = simulated["Longitude"] - geo_shift

    return simulated


def main() -> None:
    reference = pd.read_csv(REFERENCE_PATH)
    base = pd.read_csv(BASE_PATH)

    # On fabrique des donnees "de production" derivees a partir des donnees de base
    production = simulate_production_data(base)

    my_eval = generate_drift_report(reference, production, REPORT_PATH)
    summary = drift_summary(my_eval)

    print("Donnees de production simulees (inflation +20 % + decalage geo).")
    print(f"Rapport de derive : {REPORT_PATH}")
    print(
        f"Colonnes ayant derive : {int(summary['count'])} "
        f"({summary['share'] * 100:.0f}% des colonnes)"
    )


if __name__ == "__main__":
    main()
