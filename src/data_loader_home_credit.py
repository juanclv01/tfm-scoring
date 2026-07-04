"""
Carga y preprocesamiento del dataset "Home Credit Default Risk" (Kaggle).
Fuente: kaggle.com/competitions/home-credit-default-risk (fichero application_train.csv)

Rol en el TFM: prueba de escala. Con 307.511 filas y 122 columnas, el
objetivo aqui NO es maximizar el rendimiento del modelo (eso ya lo hace
German Credit como prototipo principal), sino comprobar que el pipeline
completo (preprocesado + entrenamiento + SHAP) sigue siendo viable en
tiempo y memoria cuando el volumen de datos se multiplica por 300.

Diferencia deliberada de diseno frente a data_loader.py (German Credit):
con solo 20 columnas puedes listar las categoricas a mano; con 122 no es
practico, asi que aqui se detectan automaticamente por tipo de dato.
"""
import pandas as pd

from dataset_utils import split_data

ID_COLUMN = "SK_ID_CURR"
TARGET_COLUMN = "TARGET"


def load_home_credit(path: str = "data/home-credit-default/application_train.csv") -> pd.DataFrame:
    """Carga el dataset y descarta el identificador de cliente."""
    df = pd.read_csv(path)
    df = df.rename(columns={TARGET_COLUMN: "target"})
    df = df.drop(columns=[ID_COLUMN])
    return df


def infer_feature_types(df: pd.DataFrame, target_col: str = "target"):
    """
    Deteccion automatica de columnas categoricas (dtype object) vs
    numericas (el resto). Con 121 features de entrada no es practico
    listar cada una a mano como en German Credit.
    """
    feature_cols = [c for c in df.columns if c != target_col]
    categorical = [c for c in feature_cols if df[c].dtype == "object"]
    numeric = [c for c in feature_cols if c not in categorical]
    return categorical, numeric


def build_preprocessor_home_credit(categorical_features: list, numeric_features: list):
    """
    OneHot para categoricas. Los valores nulos en columnas numericas NO
    se imputan aqui: XGBoost los maneja de forma nativa aprendiendo una
    direccion de particion por defecto para cada nodo (el algoritmo
    "sparsity-aware" de Chen & Guestrin, Seccion 3, que ya leiste).
    Imputar manualmente aqui seria redundante y podria incluso empeorar
    el resultado frente a dejar que el propio arbol decida.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder

    return ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features),
    ])


if __name__ == "__main__":
    df = load_home_credit()
    cat, num = infer_feature_types(df)
    print(f"Filas: {len(df)}, columnas totales: {len(df.columns)}")
    print(f"Categoricas detectadas: {len(cat)}, numericas detectadas: {len(num)}")
    print(f"% de valores nulos promedio por columna: {df.isnull().mean().mean():.2%}")
