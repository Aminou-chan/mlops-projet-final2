from sklearn.datasets import fetch_california_housing
import pandas as pd


def load_california_housing() -> pd.DataFrame:
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    return df
