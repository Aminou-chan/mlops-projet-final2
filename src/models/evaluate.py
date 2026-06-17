# ==================== IMPORTS ====================
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


# ==================== FONCTION DE ÉVALUATION ====================


def evaluate_model(y_true, y_pred) -> dict[str, float]:
    """
    Évalue les performances d'un modèle de régression sur un ensemble de test.

    Calcule trois métriques standards pour la régression :
    - RMSE (Root Mean Squared Error) : erreur quadratique moyenne racine
      → Pénalise fortement les grandes erreurs
      → Dans les mêmes unités que la cible

    - MAE (Mean Absolute Error) : erreur absolue moyenne
      → Plus robuste aux outliers que RMSE
      → Interprétation directe : erreur moyenne attendue

    - R² (Coefficient de détermination) : proportion de variance expliquée
      → Entre 0 et 1 idéalement (peut être négatif si pire que moyenne)
      → R² = 1 : prédictions parfaites
      → R² = 0 : modèle aussi bon que prédire la moyenne

    Args:
        y_true (array-like): Valeurs réelles de test
        y_pred (array-like): Valeurs prédites par le modèle

    Returns:
        dict[str, float]: Dictionnaire avec les 3 métriques :
            {
                'rmse': float,  # Erreur quadratique moyenne racine
                'mae': float,   # Erreur absolue moyenne
                'r2': float     # Coefficient de détermination
            }

    Example:
        >>> y_true = [3, -0.5, 2, 7]
        >>> y_pred = [2.5, 0.0, 2, 8]
        >>> metrics = evaluate_model(y_true, y_pred)
        >>> print(metrics)
        {'rmse': 0.5590, 'mae': 0.375, 'r2': 0.948}
    """
    # Calculer RMSE : √(MSE)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Calculer MAE (moyenne des erreurs absolues)
    mae = mean_absolute_error(y_true, y_pred)

    # Calculer R² (part de variance expliquée)
    r2 = r2_score(y_true, y_pred)

    # Retourner les métriques sous forme de dictionnaire avec valeurs float
    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
    }
