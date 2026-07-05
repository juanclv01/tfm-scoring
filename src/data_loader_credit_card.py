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

from dataset_utils import split_data as _split_data_generic

TARGET_CANDIDATES = [
    "default.payment.next.month",
    "default payment next month",
    "default_payment_next_month",
    "DEFAULT",
]

# Excluida por ser variable protegida (sexo), en linea con la misma
# decision aplicada a "personal_status" en German Credit (ver data_loader.py).
SENSITIVE_FEATURES_EXCLUDED = ["SEX"]

# Codificadas numericamente en el fichero original, pero son categoricas
# por naturaleza (no existe un orden significativo entre sus valores).
# NOTA: EDUCATION y MARRIAGE tienen valores fuera de la documentacion
# oficial (p.ej. EDUCATION=0,5,6; MARRIAGE=0) -- audit_data_quality()
# los detecta explicitamente en lugar de asumir silenciosamente que
# solo existen los valores documentados 1-4 / 1-3.
CATEGORICAL_FEATURES = ["EDUCATION", "MARRIAGE"]

# PAY_0/PAY_2-6 se dejan como NUMERICAS (no OneHot) de forma deliberada:
# son ordinales (a mayor valor, mas meses de retraso), y el passthrough
# preserva esa ordinalidad, mientras que OneHot la destruiria. La
# documentacion oficial de UCI solo define -1 (paga puntual) y 1-8
# (meses de retraso); los valores -2 y 0 que aparecen en los datos NO
# estan documentados (ver audit_data_quality). No se remapean sin una
# fuente oficial que confirme su significado -- se reportan, no se inventan.
NUMERIC_FEATURES = [
    "LIMIT_BAL", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]

VALORES_DOCUMENTADOS = {
    "EDUCATION": {1, 2, 3, 4},
    "MARRIAGE": {1, 2, 3},
}

# Rango documentado oficialmente por UCI para las columnas PAY_*.
# -1 y 1-8 estan definidos; -2 y 0 NO tienen definicion oficial.
PAY_COLUMNS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
PAY_VALORES_DOCUMENTADOS = {-1, 1, 2, 3, 4, 5, 6, 7, 8}


def _normalize_target_column(df: pd.DataFrame) -> pd.DataFrame:
    """El nombre de la columna objetivo varia segun la fuente de descarga."""
    for candidate in TARGET_CANDIDATES:
        if candidate in df.columns:
            return df.rename(columns={candidate: "target"})
    raise ValueError(
        f"No se encontro la columna objetivo. Se esperaba una de: {TARGET_CANDIDATES}"
    )


def load_credit_card(path: str = "data/credit-card-default/UCI_Credit_Card.csv") -> pd.DataFrame:
    """
    Carga el dataset, descarta la columna ID (no aporta señal predictiva)
    y elimina duplicados exactos -- filas identicas en TODAS las columnas
    no aportan informacion nueva y contaminarian el split train/test si
    una copia cae en cada lado.
    """
    df = pd.read_csv(path)
    df = _normalize_target_column(df)
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    n_antes = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_eliminadas = n_antes - len(df)
    if n_eliminadas > 0:
        print(f"[load_credit_card] {n_eliminadas} filas duplicadas eliminadas.")

    return df


def audit_data_quality(df: pd.DataFrame) -> dict:
    """
    Duplicados, nulos, valores fuera de rango/documentacion. Se ejecuta
    DESPUES de que load_credit_card() ya haya eliminado duplicados, por
    lo que aqui "duplicados_exactos" deberia dar 0 -- si no da 0, es que
    se llamo a audit_data_quality() sobre un df cargado sin pasar por
    load_credit_card() (p.ej. leido directamente con pd.read_csv).
    """
    n_duplicados = df.duplicated().sum()

    rangos_plausibles = {
        "AGE": (18, 100),
        "LIMIT_BAL": (0, 2_000_000),
    }
    valores_fuera_de_rango = {}
    for col, (lo, hi) in rangos_plausibles.items():
        if col in df.columns:
            fuera = df[(df[col] < lo) | (df[col] > hi)]
            if len(fuera) > 0:
                valores_fuera_de_rango[col] = len(fuera)

    codigos_no_documentados = {}
    for col, valores_validos in VALORES_DOCUMENTADOS.items():
        if col in df.columns:
            encontrados = set(df[col].unique())
            no_documentados = encontrados - valores_validos
            if no_documentados:
                codigos_no_documentados[col] = sorted(no_documentados)

    # PAY_0/PAY_2-6: mismo chequeo que EDUCATION/MARRIAGE pero sobre un
    # rango en vez de un conjunto fijo, porque -1..8 es un rango ordinal.
    pay_valores_no_documentados = {}
    for col in PAY_COLUMNS:
        if col in df.columns:
            encontrados = set(df[col].unique())
            no_documentados = encontrados - PAY_VALORES_DOCUMENTADOS
            if no_documentados:
                pay_valores_no_documentados[col] = sorted(no_documentados)

    return {
        "filas_totales": len(df),
        "duplicados_exactos": int(n_duplicados),
        "nulos_por_columna": df.isnull().sum().to_dict(),
        "valores_fuera_de_rango": valores_fuera_de_rango,
        "codigos_categoricos_no_documentados": codigos_no_documentados,
        "valores_pay_no_documentados": pay_valores_no_documentados,
    }


def build_preprocessor_credit_card():
    """Mismo patron que German Credit: OneHot para categoricas, passthrough numericas."""
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder

    return ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ("num", "passthrough", NUMERIC_FEATURES),
    ])


def split_data(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """
    Wrapper sobre dataset_utils.split_data con el nombre de la columna
    objetivo ya fijado. Permite llamar split_data(df) igual que en
    data_loader.py (German Credit), sin duplicar la logica de split.
    """
    return _split_data_generic(df, target_col="target", test_size=test_size, seed=seed)


if __name__ == "__main__":
    df = load_credit_card()
    print(f"Filas: {len(df)}, columnas: {len(df.columns)}")
    print(f"Distribucion del target:\n{df['target'].value_counts(normalize=True)}")
    print(f"\nAuditoria de calidad de datos:")
    for k, v in audit_data_quality(df).items():
        print(f"  {k}: {v}")
    print(f"\nVariables sensibles excluidas del modelo: {SENSITIVE_FEATURES_EXCLUDED}")
