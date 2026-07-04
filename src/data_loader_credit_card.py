"""
Carga y preprocesamiento del dataset "Default of Credit Card Clients" (UCI).
Fuente: archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

Rol en el TFM: validacion cruzada de metricas (AUC/Gini/KS) frente a un
segundo dataset de scoring, independiente del German Credit. Es tambien
el dataset del estudio de caso en el paper de inestabilidad de SHAP
(Risks 13(12), 2025) ya citado en la bibliografia del TFM -- conexion
directa que puedes explotar en la memoria.

Formato esperado: CSV con cabecera (variante Kaggle/UCI mas comun).
Si partes del .xls original de UCI, expórtalo antes a CSV o usa
pd.read_excel(path, header=1) para saltar la primera fila (nombres X1..X23).
"""
import pandas as pd

from dataset_utils import split_data

TARGET_CANDIDATES = [
    "default.payment.next.month",
    "default payment next month",
    "default_payment_next_month",
    "DEFAULT",
]

# Codificadas numericamente en el fichero original, pero son categoricas
# por naturaleza (no existe un orden significativo entre sus valores).
CATEGORICAL_FEATURES = ["SEX", "EDUCATION", "MARRIAGE"]

NUMERIC_FEATURES = [
    "LIMIT_BAL", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]


def _normalize_target_column(df: pd.DataFrame) -> pd.DataFrame:
    """El nombre de la columna objetivo varia segun la fuente de descarga."""
    for candidate in TARGET_CANDIDATES:
        if candidate in df.columns:
            return df.rename(columns={candidate: "target"})
    raise ValueError(
        f"No se encontro la columna objetivo. Se esperaba una de: {TARGET_CANDIDATES}"
    )


def load_credit_card(path: str = "data/credit-card-default/UCI_Credit_Card.csv") -> pd.DataFrame:
    """Carga el dataset y descarta la columna ID, que no aporta señal predictiva."""
    df = pd.read_csv(path)
    df = _normalize_target_column(df)
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
    return df


def build_preprocessor_credit_card():
    """Mismo patron que German Credit: OneHot para categoricas, passthrough numericas."""
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder

    return ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ("num", "passthrough", NUMERIC_FEATURES),
    ])


if __name__ == "__main__":
    df = load_credit_card()
    print(f"Filas: {len(df)}, columnas: {len(df.columns)}")
    print(f"Distribucion del target:\n{df['target'].value_counts(normalize=True)}")
