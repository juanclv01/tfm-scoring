"""
Carga y preprocesamiento del German Credit Dataset (UCI).
Fuente: archive.ics.uci.edu/dataset/144/statlog+german+credit+data

NOTA SOBRE VARIABLES SENSIBLES (decision de diseno documentada):
"personal_status" codifica sexo Y estado civil en una unica columna,
y la codificacion de "foreign_worker" tiene un error de documentacion
conocido en la version original de UCI (ver Ferrando et al., "Algorithmic
Fairness Datasets: the Story so Far", 2022, arXiv:2202.01711). Ambas
se EXCLUYEN deliberadamente del conjunto de features del modelo, en
linea con el Art. 10 del EU AI Act (gobernanza de datos y no discriminacion
en sistemas de IA de alto riesgo). Se mantienen en el DataFrame crudo
(load_german_credit) mas no en CATEGORICAL_FEATURES / NUMERIC_FEATURES,
por lo que el ColumnTransformer las ignora automaticamente al construir
la matriz de entrenamiento. Se mantienen tambien decodificadas en
FEATURE_VALUE_LABELS (mas abajo) solo a efectos de auditoria/documentacion
-- nunca se exponen al Nodo 3 como parte de una explicacion.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from dataset_utils import split_data as _split_data_generic

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
# en el propio codigo, no solo en la memoria.
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

# ---------------------------------------------------------------------------
# DECODIFICACION DE CODIGOS (Attribute 1-20, decode.txt / Statlog UCI).
# Fuente primaria para el Nodo 3: el LLM narrador no puede citar "A14" en
# una explicacion regulatoria -- necesita "sin cuenta corriente". Los
# diccionarios cubren TODAS las columnas cualitativas del dataset,
# incluidas personal_status y foreign_worker (excluidas del modelo, pero
# decodificadas igualmente por completitud documental y trazabilidad de
# auditoria; el Nodo 3 nunca debe recibir estas dos claves).
# ---------------------------------------------------------------------------
FEATURE_VALUE_LABELS = {
    "checking_status": {
        "A11": "cuenta corriente con saldo negativo",
        "A12": "cuenta corriente con saldo entre 0 y 200 DM",
        "A13": "cuenta corriente con saldo >= 200 DM o nomina domiciliada al menos 1 ano",
        "A14": "sin cuenta corriente",
    },
    "credit_history": {
        "A30": "sin creditos previos o todos pagados puntualmente en otras entidades",
        "A31": "todos los creditos en este banco pagados puntualmente",
        "A32": "creditos existentes pagados puntualmente hasta la fecha",
        "A33": "retraso en pagos en el pasado",
        "A34": "cuenta critica / otros creditos existentes fuera de este banco",
    },
    "purpose": {
        "A40": "coche nuevo",
        "A41": "coche usado",
        "A42": "mobiliario o equipamiento",
        "A43": "radio o television",
        "A44": "electrodomesticos",
        "A45": "reparaciones",
        "A46": "educacion",
        "A47": "vacaciones",
        "A48": "reciclaje profesional",
        "A49": "negocio",
        "A410": "otros",
    },
    "savings_status": {
        "A61": "ahorros por debajo de 100 DM",
        "A62": "ahorros entre 100 y 500 DM",
        "A63": "ahorros entre 500 y 1000 DM",
        "A64": "ahorros de 1000 DM o mas",
        "A65": "sin cuenta de ahorros o desconocido",
    },
    "employment": {
        "A71": "desempleado",
        "A72": "empleado desde hace menos de 1 ano",
        "A73": "empleado desde hace entre 1 y 4 anos",
        "A74": "empleado desde hace entre 4 y 7 anos",
        "A75": "empleado desde hace 7 anos o mas",
    },
    # EXCLUIDA DEL MODELO -- decodificada solo para auditoria.
    "personal_status": {
        "A91": "hombre divorciado o separado",
        "A92": "mujer divorciada, separada o casada",
        "A93": "hombre soltero",
        "A94": "hombre casado o viudo",
        "A95": "mujer soltera",
    },
    "other_parties": {
        "A101": "sin avalistas ni codeudores",
        "A102": "con codeudor",
        "A103": "con avalista",
    },
    "property_magnitude": {
        "A121": "propietario de bienes inmuebles",
        "A122": "seguro de vida o plan de ahorro-vivienda",
        "A123": "coche u otros bienes",
        "A124": "sin propiedades conocidas",
    },
    "other_payment_plans": {
        "A141": "planes de pago adicionales en otro banco",
        "A142": "planes de pago adicionales en comercios",
        "A143": "sin planes de pago adicionales",
    },
    "housing": {
        "A151": "vivienda en alquiler",
        "A152": "vivienda en propiedad",
        "A153": "vivienda gratuita",
    },
    "job": {
        "A171": "desempleado o no cualificado, no residente",
        "A172": "no cualificado, residente",
        "A173": "empleado cualificado o funcionario",
        "A174": "directivo, autonomo o empleado altamente cualificado",
    },
    "own_telephone": {
        "A191": "sin telefono registrado",
        "A192": "telefono registrado a nombre del cliente",
    },
    # EXCLUIDA DEL MODELO -- decodificada solo para auditoria. Codificacion
    # con error de documentacion conocido (ver docstring del modulo).
    "foreign_worker": {
        "A201": "trabajador extranjero",
        "A202": "no es trabajador extranjero",
    },
}


def load_german_credit(path: str = "data/german-credit-data/german.data") -> pd.DataFrame:
    """
    Carga el dataset original (separado por espacios, sin cabecera) y
    elimina duplicados exactos. # CORRECTED: antes esta funcion no
    deduplicaba, a diferencia de load_credit_card() y load_home_credit(),
    lo que era una inconsistencia real entre los tres loaders.
    """
    df = pd.read_csv(path, sep=" ", header=None, names=COLUMN_NAMES)
    # target original: 1=bueno, 2=malo -> remapeado a 0=bueno, 1=impago
    df["target"] = df["target"].map({1: 0, 2: 1})

    n_antes = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_eliminadas = n_antes - len(df)
    if n_eliminadas > 0:
        print(f"[load_german_credit] {n_eliminadas} filas duplicadas eliminadas.")

    return df


def audit_data_quality(df: pd.DataFrame) -> dict:
    """
    Chequeos basicos de calidad: duplicados exactos y valores numericos
    fuera de rango plausible. Tras la correccion de load_german_credit(),
    duplicados_exactos deberia dar 0 en uso normal (mismo comportamiento
    documentado en data_loader_credit_card.py).
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
    """
    Wrapper sobre dataset_utils.split_data. # CORRECTED: antes esta
    funcion reimplementaba train_test_split de forma independiente en
    vez de reutilizar dataset_utils.py, siendo la 3a copia casi identica
    de la misma logica (junto a Credit Card y Home Credit). Se elimina
    la redundancia sin cambiar la firma ni el comportamiento externo.
    """
    return _split_data_generic(df, target_col="target", test_size=test_size, seed=seed)


def parse_onehot_feature_name(shap_feature_name: str) -> tuple:
    """
    Convierte un nombre de feature transformado por el ColumnTransformer
    (p.ej. 'cat__checking_status_A14') en (nombre_original, codigo).
    Necesario porque OneHotEncoder concatena columna + categoria con un
    guion bajo, y las columnas ya contienen guiones bajos en su propio
    nombre (p.ej. 'checking_status'), por lo que un split() simple no
    basta para separar ambas partes correctamente.
    Para features numericas (prefijo 'num__') devuelve (nombre, None).
    """
    if shap_feature_name.startswith("num__"):
        return shap_feature_name.replace("num__", "", 1), None

    if shap_feature_name.startswith("cat__"):
        resto = shap_feature_name.replace("cat__", "", 1)
        for nombre_columna in FEATURE_VALUE_LABELS:
            prefijo = nombre_columna + "_"
            if resto.startswith(prefijo):
                codigo = resto[len(prefijo):]
                return nombre_columna, codigo

    # Formato no reconocido: se devuelve tal cual, sin decodificar.
    return shap_feature_name, None


def decode_feature_value(feature_name: str, code: str) -> str:
    """
    Traduce un codigo crudo (p.ej. 'A14') a su descripcion legible.
    Si la feature o el codigo no estan mapeados, devuelve el codigo
    original sin modificar -- un dato no traducible no debe romper la
    generacion del informe del Nodo 3, solo degradar la legibilidad de
    ese factor concreto.
    """
    return FEATURE_VALUE_LABELS.get(feature_name, {}).get(code, code)


# NODE_3_ENTRY_POINT: el Nodo 3 debe llamar a decode_shap_feature() sobre
# cada "feature" de state["top_features"] antes de construir el prompt
# del narrador, para no exponer nombres crudos tipo "cat__checking_status_A14"
# en la explicacion regulatoria.
def decode_shap_feature(shap_feature_name: str) -> str:
    """
    Punto de entrada principal para el Nodo 3: traduce un nombre de
    feature tal como lo devuelve el preprocesador (via SHAP) a una
    descripcion humana completa en una sola llamada.
    """
    nombre_columna, codigo = parse_onehot_feature_name(shap_feature_name)
    if codigo is None:
        return nombre_columna  # feature numerica, ya es legible (p.ej. 'age')
    return decode_feature_value(nombre_columna, codigo)


if __name__ == "__main__":
    df = load_german_credit()
    print(f"Filas: {len(df)}, columnas: {len(df.columns)}")
    print(f"Distribucion del target:\n{df['target'].value_counts(normalize=True)}")
    print(f"\nAuditoria de calidad de datos:")
    for k, v in audit_data_quality(df).items():
        print(f"  {k}: {v}")
    print(f"\nVariables sensibles excluidas del modelo: {SENSITIVE_FEATURES_EXCLUDED}")
    print(f"\nEjemplo de decodificacion: decode_shap_feature('cat__checking_status_A14') "
          f"-> '{decode_shap_feature('cat__checking_status_A14')}'")
