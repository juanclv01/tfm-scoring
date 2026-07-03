"""
Carga y preprocesamiento del German Credit Dataset (UCI).
Fuente: archive.ics.uci.edu/dataset/144/statlog+german+credit+data
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

COLUMN_NAMES = [
    "checking_status", "duration", "credit_history", "purpose",
    "credit_amount", "savings_status", "employment", "installment_rate",
    "personal_status", "other_parties", "residence_since",
    "property_magnitude", "age", "other_payment_plans", "housing",
    "existing_credits", "job", "num_dependents", "own_telephone",
    "foreign_worker", "target",
]

CATEGORICAL_FEATURES = [
    "checking_status", "credit_history", "purpose", "savings_status",
    "employment", "personal_status", "other_parties", "property_magnitude",
    "other_payment_plans", "housing", "job", "own_telephone",
    "foreign_worker",
]

NUMERIC_FEATURES = [
    "duration", "credit_amount", "installment_rate", "residence_since",
    "age", "existing_credits", "num_dependents",
]


def load_german_credit(path: str = "data/german-credit-data/german.data") -> pd.DataFrame:
    """Carga el dataset original (separado por espacios, sin cabecera)."""
    df = pd.read_csv(path, sep=" ", header=None, names=COLUMN_NAMES)
    # target original: 1=bueno, 2=malo -> remapeado a 0=bueno, 1=impago
    df["target"] = df["target"].map({1: 0, 2: 1})
    return df


def build_preprocessor() -> ColumnTransformer:
    """
    OneHot para categoricas, passthrough para numericas.
    XGBoost no requiere escalado (arboles invariantes a transf. monotonas).
    """
    return ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ("num", "passthrough", NUMERIC_FEATURES),
    ])


def split_data(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Split estratificado: el dataset tiene desbalance ~70/30."""
    X = df.drop(columns=["target"])
    y = df["target"]
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )


if __name__ == "__main__":
    df = load_german_credit()
    print(f"Filas: {len(df)}, columnas: {len(df.columns)}")
    print(f"Distribucion del target:\n{df['target'].value_counts(normalize=True)}")
