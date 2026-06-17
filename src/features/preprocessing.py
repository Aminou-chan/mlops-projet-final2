import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


TARGET = "MedHouseVal"

LOG_COLUMNS = [
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup"
]


def preprocess_data(input_path: str = "data/raw/california_housing.csv"):
    """
    Preprocess the California Housing dataset.

    Creates two pipelines:
    1. Linear pipeline:
       - train/test split
       - log transformation
       - RobustScaler

    2. Tree pipeline:
       - train/test split only (raw data)

    Saves processed datasets into data/processed/
    """

    # Load data
    df = pd.read_csv(input_path)

    # Split features / target
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]

    # Single split for both pipelines
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )


    # Pipeline 1 : Linear models
    
    X_train_linear = X_train.copy()
    X_test_linear = X_test.copy()

    # Log transformation
    for col in LOG_COLUMNS:
        X_train_linear[col] = np.log1p(X_train_linear[col])
        X_test_linear[col] = np.log1p(X_test_linear[col])

    # Robust scaling
    scaler = RobustScaler()

    X_train_linear = scaler.fit_transform(X_train_linear)
    X_test_linear = scaler.transform(X_test_linear)

    # Save scaler
    with open("models/robust_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Save linear pipeline data
    pd.DataFrame(
        X_train_linear,
        columns=X_train.columns
    ).to_csv(
        "data/processed/X_train_linear.csv",
        index=False
    )

    pd.DataFrame(
        X_test_linear,
        columns=X_test.columns
    ).to_csv(
        "data/processed/X_test_linear.csv",
        index=False
    )


    # Pipeline 2 : Tree models
    
    X_train_tree = X_train.copy()
    X_test_tree = X_test.copy()

    X_train_tree.to_csv(
        "data/processed/X_train_tree.csv",
        index=False
    )

    X_test_tree.to_csv(
        "data/processed/X_test_tree.csv",
        index=False
    )

    # Save targets
    y_train.to_csv(
        "data/processed/y_train.csv",
        index=False
    )

    y_test.to_csv(
        "data/processed/y_test.csv",
        index=False
    )

    print("Preprocessing completed successfully.")