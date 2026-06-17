"""
Module de validation des données brutes avec Great Expectations (GX Core).

Objectif:
- Définir un "contrat" (ExpectationSuite) que les données doivent respecter
- Vérifier qu'un dataset réel se conforme à ce contrat
- Détecter les anomalies (colonnes manquantes, valeurs aberrantes, etc.)

Avantages de cette approche:
- Validation déclarative et reproductible
- Détection précoce des dérives de données
- Documentation implicite de la qualité attendue

Implementation:
- Utilise un contexte GX EPHÉMÈRE (en mémoire)
- Aucun fichier/dossier de configuration n'est créé
- Les résultats sont simplement affichés
- Retourne un code de sortie 0 (succès) ou 1 (échec) pour CI/CD
"""

# ==================== IMPORTS ====================
import os
import sys

import great_expectations as gx
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

RAW_DATA_PATH = os.getenv("RAW_DATA_PATH", "data/raw/california_housing.csv")

# Colonnes attendues : 8 features + 1 cible
# Toute déviation déclenche une erreur de validation
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

# Plages réalistes pour chaque colonne
# Bornes volontairement larges pour tolérer les outliers légitimes du dataset réel
EXPECTED_RANGES = {
    "MedInc": (0, 16),  # Revenu médian en 10k$
    "HouseAge": (1, 52),  # Âge en années
    "AveRooms": (0, 150),  # Nombre moyen de pièces
    "AveBedrms": (0, 40),  # Nombre moyen de chambres
    "Population": (0, 40000),  # Population du quartier
    "AveOccup": (0, 1300),  # Taux d'occupation
    "Latitude": (32, 42),  # Latitude (Californie)
    "Longitude": (-125, -114),  # Longitude (Californie)
    "MedHouseVal": (0, 5.1),  # Prix en 100k$ (cible)
}


# ==================== FONCTIONS PRINCIPALES ====================


def build_suite(context) -> gx.ExpectationSuite:
    """
    Construit la suite d'expectations qui définit le "contrat" des données.

    Expectations (assertions):
    1. Les colonnes doivent correspondre exactement à EXPECTED_COLUMNS
    2. Aucune valeur NULL n'est tolérée dans aucune colonne
    3. Chaque colonne doit rester dans sa plage réaliste (EXPECTED_RANGES)

    Args:
        context (gx.Context): Contexte GX où ajouter la suite

    Returns:
        gx.ExpectationSuite: Suite d'expectations configurée

    Note:
        Cette suite reste entièrement en mémoire et peut être réutilisée
        pour valider différents DataFrames.
    """
    # Créer une nouvelle ExpectationSuite
    suite = context.suites.add(gx.ExpectationSuite(name="california_housing_suite"))

    # Expectation 1 : Les colonnes doivent correspondre exactement
    suite.add_expectation(
        gx.expectations.ExpectTableColumnsToMatchSet(column_set=EXPECTED_COLUMNS)
    )

    # Expectation 2 : Pas de valeurs NULL dans aucune colonne
    for column in EXPECTED_COLUMNS:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToNotBeNull(column=column)
        )

    # Expectation 3 : Valeurs dans les plages réalistes
    for column, (low, high) in EXPECTED_RANGES.items():
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeBetween(
                column=column, min_value=low, max_value=high
            )
        )

    return suite


def validate_dataframe(df: pd.DataFrame):
    """
    Valide un DataFrame contre le contrat défini (suite d'expectations).

    Processus:
    1. Créer un contexte GX éphémère (en mémoire)
    2. Ajouter le DataFrame comme source de données
    3. Construire la suite d'expectations
    4. Exécuter la validation
    5. Retourner les résultats

    Args:
        df (pd.DataFrame): DataFrame à valider

    Returns:
        gx.ValidationResult: Résultat de la validation (succès/échecs d'expectations)

    Note:
        Aucun fichier de configuration n'est créé sur le disque.
    """
    # Créer un contexte GX éphémère (en mémoire, pas de fichiers)
    context = gx.get_context()

    # Ajouter le DataFrame comme source de données
    data_source = context.data_sources.add_pandas("california_housing")

    # Créer un asset pour ce DataFrame
    data_asset = data_source.add_dataframe_asset(name="raw_data")

    # Définir un batch (partie des données) : ici le DataFrame complet
    batch_definition = data_asset.add_batch_definition_whole_dataframe("full_batch")

    # Construire la suite d'expectations
    suite = build_suite(context)

    # Obtenir un batch de données
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    # Valider le batch contre la suite et retourner les résultats
    return batch.validate(suite)


def summarize(result) -> None:
    """
    Affiche un résumé lisible et structuré des résultats de validation.

    Format de sortie:
    - Première ligne : statut global (RÉUSSIE ou ÉCHOUÉE)
    - Ensuite, une ligne par expectation :
      * [OK] si l'expectation est passée
      * [KO] si l'expectation a échoué
      * Inclut le nom de l'expectation et la colonne concernée

    Args:
        result (gx.ValidationResult): Résultat de validation à afficher

    Example:
        >>> summarize(result)
        Validation globale : RÉUSSIE
          [OK] ExpectTableColumnsToMatchSet -> (table)
          [OK] ExpectColumnValuesToNotBeNull -> MedInc
          [KO] ExpectColumnValuesToBeBetween -> Population
    """
    # Afficher le statut global
    print(f"Validation globale : {'RÉUSSIE' if result.success else 'ÉCHOUÉE'}")

    # Afficher le statut de chaque expectation
    for r in result.results:
        # Déterminer le statut (OK ou KO)
        status = "OK " if r.success else "KO "

        # Récupérer le type d'expectation
        kind = r.expectation_config.type

        # Récupérer la colonne concernée (ou "(table)" si c'est une expectation au niveau table)
        column = r.expectation_config.kwargs.get("column", "(table)")

        # Afficher la ligne
        print(f"  [{status}] {kind} -> {column}")


def main() -> None:
    """
    Point d'entrée du script.

    Processus:
    1. Charger le CSV des données brutes
    2. Valider contre le contrat
    3. Afficher les résultats
    4. Retourner un code de sortie (0 = succès, 1 = échec)

    Le code de sortie est utile pour l'automatisation (CI/CD, pipelines, etc.)
    """
    # Charger les données brutes
    df = pd.read_csv(RAW_DATA_PATH)

    # Valider le DataFrame
    result = validate_dataframe(df)

    # Afficher un résumé des résultats
    summarize(result)

    # Retourner un code de sortie non nul en cas d'échec (exploitable en CI)
    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
