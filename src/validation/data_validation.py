"""Validation des donnees brutes avec Great Expectations (GX Core).

Module autonome : il definit un "contrat" auquel le dataset California Housing
doit se conformer (colonnes, absence de valeurs manquantes, plages realistes),
puis valide un DataFrame contre ce contrat.

Utilise un contexte GX EPHEMERE (en memoire) : aucun fichier/dossier n'est cree.
"""

import os
import sys

import great_expectations as gx
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

RAW_DATA_PATH = os.getenv("RAW_DATA_PATH", "data/raw/california_housing.csv")

# Colonnes attendues (8 features + cible)
EXPECTED_COLUMNS = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
    "MedHouseVal",
]

# Plages realistes (bornes larges pour tolerer les outliers du dataset reel).
EXPECTED_RANGES = {
    "MedInc": (0, 16),
    "HouseAge": (1, 52),
    "AveRooms": (0, 150),
    "AveBedrms": (0, 40),
    "Population": (0, 40000),
    "AveOccup": (0, 1300),
    "Latitude": (32, 42),
    "Longitude": (-125, -114),
    "MedHouseVal": (0, 5.1),
}


def build_suite(context) -> gx.ExpectationSuite:
    """Construit la suite d'expectations (le 'contrat' des donnees)."""
    suite = context.suites.add(gx.ExpectationSuite(name="california_housing_suite"))

    suite.add_expectation(
        gx.expectations.ExpectTableColumnsToMatchSet(column_set=EXPECTED_COLUMNS)
    )

    for column in EXPECTED_COLUMNS:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToNotBeNull(column=column)
        )

    for column, (low, high) in EXPECTED_RANGES.items():
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeBetween(
                column=column, min_value=low, max_value=high
            )
        )

    return suite


def validate_dataframe(df: pd.DataFrame):
    """Valide un DataFrame contre le contrat. Retourne le resultat GX."""
    context = gx.get_context()  # contexte ephemere (en memoire)

    data_source = context.data_sources.add_pandas("california_housing")
    data_asset = data_source.add_dataframe_asset(name="raw_data")
    batch_definition = data_asset.add_batch_definition_whole_dataframe("full_batch")

    suite = build_suite(context)
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})
    return batch.validate(suite)


def summarize(result) -> None:
    """Affiche un resume lisible des resultats de validation."""
    print(f"Validation globale : {'REUSSIE' if result.success else 'ECHOUEE'}")
    for r in result.results:
        status = "OK " if r.success else "KO "
        kind = r.expectation_config.type
        column = r.expectation_config.kwargs.get("column", "(table)")
        print(f"  [{status}] {kind} -> {column}")


def main() -> None:
    df = pd.read_csv(RAW_DATA_PATH)
    result = validate_dataframe(df)
    summarize(result)
    # code de sortie non nul si echec -> exploitable en CI
    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
