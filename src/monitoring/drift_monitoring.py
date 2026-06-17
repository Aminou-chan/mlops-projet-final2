"""Detection de derive (data drift) avec Evidently.

Compare un jeu de donnees de reference (l'entrainement) a un jeu de donnees
courant (de nouvelles donnees / la production) pour detecter si la distribution
des features a change. Une derive importante est un signal que le modele
pourrait se degrader et qu'un re-entrainement est a envisager.

Genere un rapport HTML interactif dans reports/drift/.
"""

import os

import pandas as pd
from dotenv import load_dotenv
from evidently import Report
from evidently.presets import DataDriftPreset

load_dotenv()

REFERENCE_PATH = os.getenv("DRIFT_REFERENCE_PATH", "data/processed/X_train_tree.csv")
CURRENT_PATH = os.getenv("DRIFT_CURRENT_PATH", "data/processed/X_test_tree.csv")
REPORT_PATH = os.getenv("DRIFT_REPORT_PATH", "reports/drift/data_drift_report.html")


def build_report(reference: pd.DataFrame, current: pd.DataFrame):
    """Construit et execute le rapport de derive (sans sauvegarde)."""
    report = Report([DataDriftPreset()], include_tests=True)
    # ordre attendu par Evidently : (donnees courantes, donnees de reference)
    return report.run(current, reference)


def drift_summary(my_eval) -> dict:
    """Extrait le nombre et la part de colonnes ayant derive."""
    for metric in my_eval.dict()["metrics"]:
        if "DriftedColumnsCount" in metric.get("metric_name", ""):
            return metric["value"]  # {"count": ..., "share": ...}
    return {"count": 0, "share": 0.0}


def generate_drift_report(
    reference: pd.DataFrame, current: pd.DataFrame, output_path: str = REPORT_PATH
):
    """Genere le rapport de derive et sauvegarde le HTML interactif."""
    my_eval = build_report(reference, current)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    my_eval.save_html(output_path)
    return my_eval


def main() -> None:
    reference = pd.read_csv(REFERENCE_PATH)
    current = pd.read_csv(CURRENT_PATH)

    my_eval = generate_drift_report(reference, current)
    summary = drift_summary(my_eval)

    print(f"Rapport de derive genere : {REPORT_PATH}")
    print(
        f"Colonnes ayant derive : {int(summary['count'])} "
        f"({summary['share'] * 100:.0f}% des colonnes)"
    )


if __name__ == "__main__":
    main()
