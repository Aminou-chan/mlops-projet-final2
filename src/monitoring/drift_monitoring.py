"""
Module de détection de data drift (dérive de données) avec Evidently.

Objectif:
- Détecter si les distributions des features ont changé au fil du temps
- Comparer les données de référence (entraînement) aux données courantes (production)
- Identifier quelles features ont dérivé et de combien
- Générer des rapports HTML interactifs

Contexte:
- En production, les données peuvent s'éloigner de celles vues à l'entraînement
- Exemples : inflation économique, changement démographique, évolution légale, etc.
- Une dérive significative peut mener à une dégradation du modèle
- D'où l'importance de détecter et re-entraîner régulièrement

Evidently:
- Analyse statistique des distributions
- Détecte les dérives avec des tests statistiques
- Génère des rapports HTML interactifs pour visualisation
"""

# ==================== IMPORTS ====================
import os

import pandas as pd
from dotenv import load_dotenv
from evidently import Report
from evidently.presets import DataDriftPreset

# Charger les variables d'environnement
load_dotenv()

# ==================== CONFIGURATION ====================
# Chemin du dataset de référence (données d'entraînement)
# C'est la distribution "de base" contre laquelle on comparera
REFERENCE_PATH = os.getenv("DRIFT_REFERENCE_PATH", "data/processed/X_train_tree.csv")

# Chemin du dataset courant (données nouvelles / production)
# C'est celui qu'on va comparer au référence pour détecter la dérive
CURRENT_PATH = os.getenv("DRIFT_CURRENT_PATH", "data/processed/X_test_tree.csv")

# Chemin de sortie du rapport de dérive (HTML interactif)
REPORT_PATH = os.getenv("DRIFT_REPORT_PATH", "reports/drift/data_drift_report.html")


# ==================== FONCTIONS ====================


def build_report(reference: pd.DataFrame, current: pd.DataFrame):
    """
    Construit et exécute un rapport de dérive de données.

    Utilise Evidently's DataDriftPreset pour :
    - Analyser les distributions de chaque feature
    - Comparer les statistiques descriptives
    - Appliquer des tests statistiques pour détecter les dérives
    - Inclure des tests de validation

    Args:
        reference (pd.DataFrame): Dataset de référence (entraînement)
        current (pd.DataFrame): Dataset courant (production / test)

    Returns:
        evidently.Report: Rapport contenant les résultats de l'analyse

    Note:
        L'ordre des paramètres est important : (current, reference)
        Evidently compare le courant par rapport au référence
    """
    # Créer un rapport avec le preset DataDrift et inclure les tests
    report = Report([DataDriftPreset()], include_tests=True)

    # Exécuter l'analyse (attention à l'ordre : courant d'abord, puis référence)
    return report.run(current, reference)


def drift_summary(my_eval) -> dict:
    """
    Extrait le résumé de la dérive depuis le rapport.

    Récupère :
    - count : nombre de features qui ont dérivé
    - share : proportion de features qui ont dérivé (0.0 à 1.0)

    Args:
        my_eval (evidently.Report): Rapport généré par Evidently

    Returns:
        dict: {"count": int, "share": float}
               Retourne {"count": 0, "share": 0.0} si non trouvé

    Example:
        >>> summary = drift_summary(report)
        >>> print(summary)
        {'count': 3, 'share': 0.375}  # 3 colonnes / 8 = 37.5%
    """
    # Parcourir les métriques du rapport
    for metric in my_eval.dict()["metrics"]:
        # Chercher la métrique "DriftedColumnsCount"
        if "DriftedColumnsCount" in metric.get("metric_name", ""):
            # Retourner sa valeur (dictionnaire avec count et share)
            return metric["value"]

    # Si pas trouvé, retourner un résumé par défaut (pas de dérive)
    return {"count": 0, "share": 0.0}


def generate_drift_report(
    reference: pd.DataFrame, current: pd.DataFrame, output_path: str = REPORT_PATH
):
    """
    Génère un rapport de dérive complet et le sauvegarde en HTML.

    Processus:
    1. Construit le rapport d'analyse de dérive
    2. Crée le répertoire de sortie s'il n'existe pas
    3. Sauvegarde le rapport en HTML interactif
    4. Retourne le rapport pour traitement ultérieur

    Args:
        reference (pd.DataFrame): Dataset de référence (entraînement)
        current (pd.DataFrame): Dataset courant (production / test)
        output_path (str): Chemin de sortie du fichier HTML (défaut: REPORT_PATH)

    Returns:
        evidently.Report: Rapport d'analyse de dérive

    Side effects:
        - Crée le répertoire parent du output_path s'il n'existe pas
        - Écrit un fichier HTML à output_path
    """
    # Construire et exécuter l'analyse
    my_eval = build_report(reference, current)

    # Créer les répertoires parents s'ils n'existent pas
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Sauvegarder le rapport en HTML interactif
    my_eval.save_html(output_path)

    # Retourner le rapport pour extraction de métadonnées
    return my_eval


# ==================== FONCTION PRINCIPALE ====================


def main() -> None:
    """
    Point d'entrée du module : effectue l'analyse de dérive complète.

    Processus:
    1. Charger le dataset de référence
    2. Charger le dataset courant
    3. Générer le rapport de dérive
    4. Afficher un résumé textuel
    """
    # Charger le dataset de référence (entraînement)
    reference = pd.read_csv(REFERENCE_PATH)

    # Charger le dataset courant (production / test)
    current = pd.read_csv(CURRENT_PATH)

    # Générer le rapport et sauvegarder en HTML
    my_eval = generate_drift_report(reference, current)

    # Extraire le résumé de dérive
    summary = drift_summary(my_eval)

    # Afficher les résultats
    print(f"Rapport de derive genere : {REPORT_PATH}")
    print(
        f"Colonnes ayant derive : {int(summary['count'])} "
        f"({summary['share'] * 100:.0f}% des colonnes)"
    )


if __name__ == "__main__":
    main()
