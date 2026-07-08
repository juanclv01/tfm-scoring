"""
Carga y preprocesamiento del dataset "Default of Credit Card Clients" (UCI).
Fuente: archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

Rol en el TFM: validacion cruzada de metricas (AUC/Gini/KS) frente a un
segundo dataset de scoring, independiente del German Credit. Es tambien
el dataset del estudio de caso en el paper de inestabilidad de SHAP
(Risks 13(12), 2025) ya citado en la bibliografia del TFM.

Formato de entrada: el .xls original de UCI (no CSV). load_credit_card()
acepta tanto una RUTA A CARPETA (busca el .xls/.xlsx dentro automaticamente,
sin depender de conocer el nombre exacto del fichero) como una ruta directa
a un fichero concreto.
"""
import glob
import os

import pandas as pd

from dataset_utils import split_data as _split_data_generic

TARGET_CANDIDATES = [
    "default.payment.next.month",
    "default payment next month",
    "default_payment_next_month",
    "DEFAULT",
]

SENSITIVE_FEATURES_EXCLUDED = ["SEX"]

CATEGORICAL_FEATURES = ["EDUCATION", "MARRIAGE"]

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

PAY_COLUMNS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
PAY_VALORES_DOCUMENTADOS = {-1, 1, 2, 3, 4, 5, 6, 7, 8}


def _resolve_data_file(path: str) -> str:
    """
    Si 'path' es una carpeta, busca dentro el primer .xls/.xlsx y devuelve
    su ruta completa. Si 'path' ya es un fichero, lo devuelve tal cual.

    # CORRECTED: evita depender de conocer el nombre exacto del fichero
    descargado (UCI, Kaggle y reexportaciones manuales lo nombran de forma
    distinta), ya que el usuario solo indico la CARPETA de destino, no un
    nombre de archivo concreto.
    """
    if os.path.isdir(path):
        candidatos = sorted(
            glob.glob(os.path.join(path, "*.xls"))
            + glob.glob(os.path.join(path, "*.xlsx"))
        )
        if not candidatos:
            raise FileNotFoundError(
                f"No se encontro ningun fichero .xls/.xlsx dentro de: {path}"
            )
        if len(candidatos) > 1:
            print(f"[load_credit_card] Aviso: varios ficheros Excel encontrados "
                  f"en '{path}'; usando el primero por orden alfabetico: "
                  f"{os.path.basename(candidatos[0])}")
        return candidatos[0]

    if not os.path.isfile(path):
        raise FileNotFoundError(f"No se encontro el fichero: {path}")
    return path


def _read_raw_file(file_path: str) -> pd.DataFrame:
    """
    El .xls/.xlsx original de UCI tiene DOS filas de cabecera: la fila 0
    trae las etiquetas genericas del paper (ID, X1, X2... Y); la fila 1
    trae los nombres reales de columna (LIMIT_BAL, SEX, EDUCATION...).
    Por eso header=1 (saltar fila 0, usar fila 1 como cabecera real) --
    verificado con un fichero de prueba que replica esta estructura exacta
    antes de integrar esta funcion. El CSV (variante Kaggle, si se usara
    en el futuro) ya trae una unica fila de cabecera correcta.

    Requiere 'xlrd' instalado para leer .xls legacy, u 'openpyxl' para
    .xlsx (ver requirements.txt).
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext in (".xls", ".xlsx"):
        return pd.read_excel(file_path, header=1)
    return pd.read_csv(file_path)


def _normalize_target_column(df: pd.DataFrame) -> pd.DataFrame:
    """El nombre de la columna objetivo varia segun la fuente de descarga."""
    for candidate in TARGET_CANDIDATES:
        if candidate in df.columns:
            return df.rename(columns={candidate: "target"})
    raise ValueError(
        f"No se encontro la columna objetivo. Se esperaba una de: {TARGET_CANDIDATES}. "
        f"Columnas encontradas: {list(df.columns)}"
    )


def load_credit_card(path: str = "data/default-of-credit-card-clients") -> pd.DataFrame:
    """
    Carga el dataset (acepta ruta a carpeta o a fichero concreto),
    descarta la columna ID y elimina duplicados exactos.
    """
    file_path = _resolve_data_file(path)
    df = _read_raw_file(file_path)
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
    lo que "duplicados_exactos" deberia dar 0 en uso normal.
    """
    n_duplicados = df.duplicated().sum()

    rangos_plausibles = {"AGE": (18, 100), "LIMIT_BAL": (0, 2_000_000)}
    valores_fuera_de_rango = {}
    for col, (lo, hi) in rangos_plausibles.items():
        if col in df.columns:
            fuera = df[(df[col] < lo) | (df[col] > hi)]
            if len(fuera) > 0:
                valores_fuera_de_rango[col] = len(fuera)

    codigos_no_documentados = {}
    for col, valores_validos in VALORES_DOCUMENTADOS.items():
        if col in df.columns:
            no_doc = set(df[col].unique()) - valores_validos
            if no_doc:
                codigos_no_documentados[col] = sorted(no_doc)

    pay_no_documentados = {}
    for col in PAY_COLUMNS:
        if col in df.columns:
            no_doc = set(df[col].unique()) - PAY_VALORES_DOCUMENTADOS
            if no_doc:
                pay_no_documentados[col] = sorted(no_doc)

    return {
        "filas_totales": len(df),
        "duplicados_exactos": int(n_duplicados),
        "nulos_por_columna": df.isnull().sum().to_dict(),
        "valores_fuera_de_rango": valores_fuera_de_rango,
        "codigos_categoricos_no_documentados": codigos_no_documentados,
        "valores_pay_no_documentados": pay_no_documentados,
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
    """Wrapper sobre dataset_utils.split_data, mismo patron que en los otros loaders."""
    return _split_data_generic(df, target_col="target", test_size=test_size, seed=seed)


if __name__ == "__main__":
    df = load_credit_card()
    print(f"Filas: {len(df)}, columnas: {len(df.columns)}")
    print(f"Distribucion del target:\n{df['target'].value_counts(normalize=True)}")
    print(f"\nAuditoria de calidad de datos:")
    for k, v in audit_data_quality(df).items():
        print(f"  {k}: {v}")
    print(f"\nVariables sensibles excluidas del modelo: {SENSITIVE_FEATURES_EXCLUDED}")
