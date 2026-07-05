"""
Carga y preprocesamiento del dataset "Home Credit Default Risk" (Kaggle).
Fuente: kaggle.com/competitions/home-credit-default-risk (fichero application_train.csv)

Rol en el TFM: prueba de escala. Con 307.511 filas y 122 columnas, el
objetivo aqui NO es maximizar el rendimiento del modelo (eso ya lo hace
German Credit como prototipo principal), sino comprobar que el pipeline
completo (preprocesado + entrenamiento + SHAP) sigue siendo viable en
tiempo y memoria cuando el volumen de datos se multiplica por 300.
"""
import numpy as np
import pandas as pd

from dataset_utils import split_data as _split_data_generic

ID_COLUMN = "SK_ID_CURR"
TARGET_COLUMN = "TARGET"

# CODE_GENDER es la columna de sexo/genero de este dataset (valores
# 'M', 'F', y 4 filas con 'XNA' -- confirmado empiricamente sobre las
# 307.511 filas). Se excluye del modelo por el mismo motivo que
# personal_status en German Credit y SEX en Credit Card: variable
# protegida, Art. 10 EU AI Act. audit_data_quality() reporta el conteo
# de 'XNA' solo con fines de documentacion -- no afecta al modelo,
# porque la columna ya no forma parte de sus features en ningun caso.
SENSITIVE_FEATURES_EXCLUDED = ["CODE_GENDER"]

# DAYS_EMPLOYED usa 365243 como placeholder de "no empleado / no aplica"
# en lugar de un nulo -- un valor fisicamente imposible (equivale a mas
# de 1000 años de antigüedad laboral). Es la anomalia mas documentada de
# este dataset. En lugar de corregir solo esta columna, load_home_credit()
# busca el MISMO numero magico en cualquier columna numerica: si es un
# placeholder sistematico del formato del dataset, podria aparecer en
# mas sitios de los que la literatura menciona explicitamente.
VALOR_PLACEHOLDER_ANOMALO = 365243


def load_home_credit(path: str = "data/home-credit-default/application_train.csv") -> pd.DataFrame:
    """
    Carga el dataset, descarta el ID, elimina duplicados exactos y
    corrige el placeholder anomalo (365243) alli donde aparezca.
    """
    df = pd.read_csv(path)
    df = df.rename(columns={TARGET_COLUMN: "target"})
    df = df.drop(columns=[ID_COLUMN])

    n_antes = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_eliminadas = n_antes - len(df)
    if n_eliminadas > 0:
        print(f"[load_home_credit] {n_eliminadas} filas duplicadas eliminadas.")

    columnas_numericas = df.select_dtypes(include=[np.number]).columns
    columnas_afectadas = []
    for col in columnas_numericas:
        if (df[col] == VALOR_PLACEHOLDER_ANOMALO).any():
            df[col] = df[col].replace(VALOR_PLACEHOLDER_ANOMALO, np.nan)
            columnas_afectadas.append(col)
    if columnas_afectadas:
        print(f"[load_home_credit] Placeholder {VALOR_PLACEHOLDER_ANOMALO} "
              f"corregido (-> NaN) en: {columnas_afectadas}")

    return df


def audit_data_quality(df: pd.DataFrame, target_col: str = "target") -> dict:
    """
    Mismo espiritu que en German Credit y Credit Card. Con 122 columnas
    no se listan rangos plausibles uno a uno; se reporta el porcentaje
    de nulos por columna, duplicados, la anomalia de CODE_GENDER='XNA'
    (informativa, no afecta al modelo) y el extremo superior de
    AMT_INCOME_TOTAL (se reporta, NO se recorta automaticamente: un
    ingreso muy alto es infrecuente pero no imposible, a diferencia de
    365243 dias trabajados).
    """
    n_duplicados = df.duplicated().sum()
    nulos_pct = (df.isnull().mean() * 100).round(1)
    columnas_con_nulos = nulos_pct[nulos_pct > 0].sort_values(ascending=False)

    xna_genero = None
    if "CODE_GENDER" in df.columns:
        xna_genero = int((df["CODE_GENDER"] == "XNA").sum())

    ingreso_extremo = {}
    if "AMT_INCOME_TOTAL" in df.columns:
        ingreso_extremo = {
            "percentil_99": float(df["AMT_INCOME_TOTAL"].quantile(0.99)),
            "maximo": float(df["AMT_INCOME_TOTAL"].max()),
        }

    return {
        "filas_totales": len(df),
        "columnas_totales": len(df.columns),
        "duplicados_exactos": int(n_duplicados),
        "columnas_con_nulos": len(columnas_con_nulos),
        "top_5_columnas_mas_nulos_pct": columnas_con_nulos.head(5).to_dict(),
        "code_gender_xna_filas": xna_genero,
        "amt_income_total_extremos": ingreso_extremo,
    }


def infer_feature_types(df: pd.DataFrame, target_col: str = "target"):
    """
    Deteccion automatica de columnas categoricas (dtype object) vs
    numericas, EXCLUYENDO explicitamente las variables sensibles.
    Con 121 features de entrada no es practico listar cada una a mano
    como en German Credit, pero la exclusion de variables protegidas
    no puede depender de un descarte automatico -- se hace explicita.
    """
    feature_cols = [
        c for c in df.columns
        if c != target_col and c not in SENSITIVE_FEATURES_EXCLUDED
    ]
    categorical = [c for c in feature_cols if df[c].dtype == "object"]
    numeric = [c for c in feature_cols if c not in categorical]
    return categorical, numeric


def build_preprocessor_home_credit(categorical_features: list, numeric_features: list):
    """
    OneHot para categoricas. Los valores nulos en columnas numericas NO
    se imputan aqui: XGBoost los maneja de forma nativa aprendiendo una
    direccion de particion por defecto para cada nodo (el algoritmo
    "sparsity-aware" de Chen & Guestrin, Seccion 3, que ya leiste).
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder

    return ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features),
    ])


def split_data(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """
    Wrapper sobre dataset_utils.split_data con el nombre de la columna
    objetivo ya fijado, igual que en data_loader.py y data_loader_credit_card.py.
    """
    return _split_data_generic(df, target_col="target", test_size=test_size, seed=seed)


if __name__ == "__main__":
    df = load_home_credit()
    cat, num = infer_feature_types(df)
    print(f"Filas: {len(df)}, columnas totales: {len(df.columns)}")
    print(f"Categoricas detectadas (tras excluir sensibles): {len(cat)}")
    print(f"Numericas detectadas: {len(num)}")
    print(f"Variables sensibles excluidas del modelo: {SENSITIVE_FEATURES_EXCLUDED}")
    print(f"\nAuditoria de calidad de datos:")
    for k, v in audit_data_quality(df).items():
        print(f"  {k}: {v}")
