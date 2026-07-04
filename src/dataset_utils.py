"""
Utilidades compartidas entre los distintos loaders de datasets.

data_loader.py (German Credit) mantiene su propio split_data() por
compatibilidad con el codigo ya distribuido; los loaders nuevos
(Home Credit, Credit Card) reutilizan esta version generica para no
triplicar la misma logica de 5 lineas tres veces.
"""
import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(df: pd.DataFrame, target_col: str, test_size: float = 0.2, seed: int = 42):
    """Split estratificado generico, valido para cualquier dataset binario."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
