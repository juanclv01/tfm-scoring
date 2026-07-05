"""
Carga y preprocesamiento del German Credit Dataset (UCI).
Fuente: archive.ics.uci.edu/dataset/144/statlog+german+credit+data

NOTA SOBRE VARIABLES SENSIBLES (decisión de diseño documentada):
"personal_status" codifica sexo Y estado civil en una única columna,
y la codificación de "foreign_worker" tiene un error de documentacion
conocido en la version original de UCI (ver Ferrando et al., "Algorithmic
Fairness Datasets: the Story so Far", 2022, arXiv:2202.01711). 

Ambas se EXCLUYEN deliberadamente del conjunto de features del modelo, en
linea con el Art. 10 del EU AI Act (gobernanza de datos y no discriminacion
en sistemas de IA de alto riesgo). Se mantienen en el DataFrame crudo
(load_german_credit) mas no en CATEGORICAL_FEATURES / NUMERIC_FEATURES,
por lo que el ColumnTransformer las ignora automaticamente al construir
la matriz de entrenamiento.
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

# Excluidas del modelo por ser variables protegidas / con error de
# documentacion conocido. Se listan aqui explicitamente (en vez de solo
# omitirlas) para que la exclusion sea auditable y quede documentada
# en el propio código, no solo en la memoria.
SENSITIVE_FEATURES_EXCLUDED = ["personal_status", "foreign_worker"]

CATEGORICAL_FEATURES = [
    "checking_status", "credit_history", "purpose", "savings_status",
    "employment", "other_parties", "property_magnitude",
    "other_payment_plans", "housing", "job", "own_telephone",
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


def audit_data_quality(df: pd.DataFrame) -> dict:
    """
    Preproceso de auditoria de calidad de datos: nulos,
    duplicados exactos y valores numericos fuera de rango plausible.
    No elimina nada automaticamente -- informa, para que la decision de
    que hacer con cada hallazgo quede documentada y sea deliberada.
    """
    n_duplicados = df.duplicated().sum()

    rangos_plausibles = {
        "age": (18, 100),
        "duration": (1, 120),        # meses
        "credit_amount": (0, 200_000),
    }
    valores_fuera_de_rango = {}
    for col, (lo, hi) in rangos_plausibles.items():
        if col in df.columns:
            fuera = df[(df[col] < lo) | (df[col] > hi)]
            if len(fuera) > 0:
                valores_fuera_de_rango[col] = len(fuera)

    return {
        "filas_totales": len(df),
        "duplicados_exactos": int(n_duplicados),
        "nulos_por_columna": df.isnull().sum().to_dict(),
        "valores_fuera_de_rango": valores_fuera_de_rango,
    }


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
    print(f"\nAuditoria de calidad de datos:")
    for k, v in audit_data_quality(df).items():
        print(f"  {k}: {v}")
    print(f"\nVariables sensibles excluidas del modelo: {SENSITIVE_FEATURES_EXCLUDED}")
