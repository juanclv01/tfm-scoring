"""
Carga y preprocesamiento del German Credit Dataset (UCI).
Fuente: archive.ics.uci.edu/dataset/144/statlog+german+credit+data

NOTA SOBRE VARIABLES SENSIBLES (decision de diseno documentada):
"personal_status" codifica sexo Y estado civil en una unica columna,
y la codificacion de "foreign_worker" tiene un error de documentacion
conocido en la version original de UCI (ver Ferrando et al., "Algorithmic
Fairness Datasets: the Story so Far", 2022, arXiv:2202.01711). Ambas
se EXCLUYEN deliberadamente del conjunto de features del modelo, en
linea con el Art. 10 del EU AI Act.
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
# DECODIFICACION DE VALORES CATEGORICOS (decode.txt / Statlog UCI)
# Fuente primaria para el NARRATOR: sustituye los codigos crudos (A11, A32...)
# por su significado real en espanol.
# personal_status y foreign_worker se incluyen solo a efectos de auditoria
# -- el NARRATOR NUNCA debe recibirlas como features activas.
# ---------------------------------------------------------------------------
FEATURE_VALUE_LABELS = {
    "checking_status": {
        "A11": "saldo negativo en cuenta corriente",
        "A12": "saldo entre 0 y 200 DM en cuenta corriente",
        "A13": "saldo de 200 DM o mas en cuenta corriente, o nomina domiciliada al menos 1 ano",
        "A14": "sin cuenta corriente",
    },
    "credit_history": {
        "A30": "sin creditos previos o todos pagados puntualmente en otras entidades",
        "A31": "todos los creditos en este banco pagados puntualmente",
        "A32": "creditos existentes pagados puntualmente hasta la fecha",
        "A33": "retraso en pagos registrado en el pasado",
        "A34": "cuenta critica o creditos existentes en otras entidades",
    },
    "purpose": {
        "A40": "compra de coche nuevo",
        "A41": "compra de coche usado",
        "A42": "compra de mobiliario o equipamiento del hogar",
        "A43": "compra de television o radio",
        "A44": "compra de electrodomesticos",
        "A45": "reparaciones del hogar",
        "A46": "educacion o formacion",
        "A47": "vacaciones",
        "A48": "reciclaje o formacion profesional",
        "A49": "negocio o actividad empresarial",
        "A410": "otros propositos",
    },
    "savings_status": {
        "A61": "ahorros inferiores a 100 DM",
        "A62": "ahorros entre 100 y 500 DM",
        "A63": "ahorros entre 500 y 1000 DM",
        "A64": "ahorros de 1000 DM o mas",
        "A65": "sin cuenta de ahorros o importe desconocido",
    },
    "employment": {
        "A71": "desempleado",
        "A72": "empleado desde hace menos de 1 ano",
        "A73": "empleado entre 1 y 4 anos",
        "A74": "empleado entre 4 y 7 anos",
        "A75": "empleado desde hace 7 anos o mas",
    },
    "personal_status": {  # EXCLUIDA DEL MODELO -- solo auditoria
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
        "A123": "propietario de coche u otros bienes",
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
        "A172": "trabajador no cualificado residente",
        "A173": "empleado cualificado o funcionario",
        "A174": "directivo, autonomo o empleado altamente cualificado",
    },
    "own_telephone": {
        "A191": "sin telefono registrado a su nombre",
        "A192": "telefono registrado a nombre del solicitante",
    },
    "foreign_worker": {  # EXCLUIDA DEL MODELO -- solo auditoria
        "A201": "trabajador extranjero",
        "A202": "no es trabajador extranjero",
    },
}

# ---------------------------------------------------------------------------
# TRADUCCIONES AL ESPANOL DE LOS NOMBRES DE VARIABLES
# Cubre tanto las categoricas como las numericas.
# El NARRATOR recibe estos nombres en espanol para que la narrativa
# generada sea directamente comprensible para el cliente final.
# ---------------------------------------------------------------------------
FEATURE_NAME_LABELS_ES = {
    # Categoricas (activas en el modelo)
    "checking_status":      "Estado de la cuenta corriente",
    "credit_history":       "Historial crediticio",
    "purpose":              "Proposito del credito",
    "savings_status":       "Estado de la cuenta de ahorros",
    "employment":           "Situacion laboral",
    "other_parties":        "Avalistas o codeudores",
    "property_magnitude":   "Propiedades del solicitante",
    "other_payment_plans":  "Otros planes de pago activos",
    "housing":              "Tipo de vivienda",
    "job":                  "Categoria profesional",
    "own_telephone":        "Telefono registrado",
    # Numericas
    "duration":             "Duracion del prestamo (meses)",
    "credit_amount":        "Importe del credito (DM)",
    "installment_rate":     "Cuota como porcentaje de ingresos",
    "residence_since":      "Anos en la residencia actual",
    "age":                  "Edad del solicitante (anos)",
    "existing_credits":     "Numero de creditos activos en este banco",
    "num_dependents":       "Numero de personas a cargo",
    # Excluidas del modelo -- se incluyen solo para trazabilidad
    "personal_status":      "Estado civil y sexo (EXCLUIDA)",
    "foreign_worker":       "Trabajador extranjero (EXCLUIDA)",
}

# ---------------------------------------------------------------------------
# FORMATO DE VALORES NUMERICOS PARA EL NARRATOR
# Permite presentar los valores numericos con contexto y unidades,
# no como cifras aisladas. El template usa {value} como placeholder.
#
# NOTA (pendiente de revisar en otra sesion, fuera del alcance de esta
# limpieza de ingesta/modelo): "installment_rate" en Statlog UCI documenta
# el Attribute 8 como valores discretos 1-4, probablemente un tramo/bracket
# y no un porcentaje continuo. El template actual "{value}% de los ingresos
# netos" puede sugerir una precision que el dato no tiene. No se toca aqui
# porque afecta al Nodo 3 (Narrator), fuera del alcance de esta revision
# centrada en ingesta y en el modelo XGBoost.
# ---------------------------------------------------------------------------
NUMERIC_FEATURE_FORMAT = {
    "duration":          "{value} meses",
    "credit_amount":     "{value} DM",
    "installment_rate":  "{value}% de los ingresos netos",
    "residence_since":   "{value} anos",
    "age":               "{value} anos",
    "existing_credits":  "{value} credito(s)",
    "num_dependents":    "{value} persona(s) a cargo",
}


# ---------------------------------------------------------------------------
# FUNCIONES DE DECODIFICACION PARA EL NARRATOR
# ---------------------------------------------------------------------------

def _format_numeric_value(feature_name: str, value) -> str:
    """Formatea un valor numerico aplicando el template de unidades."""
    template = NUMERIC_FEATURE_FORMAT.get(feature_name, "{value}")
    try:
        return template.format(value=int(value) if float(value) == int(float(value)) else round(float(value), 2))
    except (ValueError, TypeError):
        return str(value)


def parse_onehot_feature_name(shap_feature_name: str) -> tuple:
    """
    Convierte un nombre de feature transformado por ColumnTransformer
    (p.ej. 'cat__checking_status_A14') en (nombre_columna, codigo).
    Para features numericas (prefijo 'num__') devuelve (nombre, None).
    Usa la lista conocida de nombres de columna para separar correctamente
    nombre de columna y codigo, evitando el problema de los guiones bajos
    dentro del propio nombre de columna (p.ej. 'checking_status').
    """
    if shap_feature_name.startswith("num__"):
        return shap_feature_name.replace("num__", "", 1), None

    if shap_feature_name.startswith("cat__"):
        resto = shap_feature_name.replace("cat__", "", 1)
        for nombre_columna in FEATURE_VALUE_LABELS:
            if resto.startswith(nombre_columna + "_"):
                codigo = resto[len(nombre_columna) + 1:]
                return nombre_columna, codigo

    return shap_feature_name, None


def decode_feature_value(feature_name: str, code: str) -> str:
    """
    Traduce un codigo categorico crudo (p.ej. 'A14') a su descripcion
    en espanol. Si no se encuentra, devuelve el codigo original sin modificar
    para no romper la generacion del informe del NARRATOR.
    """
    return FEATURE_VALUE_LABELS.get(feature_name, {}).get(code, code)


def decode_shap_feature(shap_feature_name: str) -> str:
    """
    Devuelve el nombre de la feature en espanol a partir del nombre
    transformado por el preprocesador.
    - 'cat__checking_status_A14' -> 'Estado de la cuenta corriente'
    - 'num__age' -> 'Edad del solicitante (anos)'
    """
    nombre_columna, _ = parse_onehot_feature_name(shap_feature_name)
    return FEATURE_NAME_LABELS_ES.get(nombre_columna, nombre_columna)


def build_narrator_tuple(
    shap_feature_name: str,
    client_data: dict,
    shap_value: float,
) -> dict:
    """
    Construye la tupla completa que necesita el NARRATOR segun el formato
    de Explingo (Zytek et al., 2024):
        (feature_name, feature_value, SHAP contribution)

    donde feature_name y feature_value estan en espanol y son directamente
    legibles por el cliente final sin necesidad de conocer los codigos de UCI.

    Para features categoricas: extrae el valor real del cliente desde
    client_data y lo decodifica a su descripcion en espanol.
    Para features numericas: formatea el valor numerico con sus unidades.

    Args:
        shap_feature_name: nombre de la feature en el espacio del
                           preprocesador (p.ej. 'cat__checking_status_A14').
        client_data:       diccionario con los datos crudos del cliente
                           (las mismas claves que COLUMN_NAMES, sin 'target').
        shap_value:        contribucion de esta feature para este cliente,
                           YA CONVERTIDA a PUNTOS DE SCORE por node_explainability
                           (Nodo 2) antes de llamar a esta funcion -- no es la
                           salida cruda del explainer, que esta en espacio de
                           proba_default. Convencion: shap_value > 0 -> SUBE
                           el score; shap_value < 0 -> BAJA el score. El
                           NARRATOR debe usar directamente este signo (ya
                           coincide con la lectura intuitiva "positivo = mejora").

    Returns:
        Diccionario con las tres claves del formato Explingo:
            feature_name    (str, en espanol)
            feature_value   (str, legible y en espanol)
            shap_value      (float, PUNTOS DE SCORE, no probabilidad;
                            signo ya listo para el NARRATOR)
    """
    nombre_columna, codigo_ohe = parse_onehot_feature_name(shap_feature_name)

    # CORRECTED: antes se repetia aqui el mismo lookup
    # (FEATURE_NAME_LABELS_ES.get(nombre_columna, nombre_columna)) que ya
    # hace decode_shap_feature(). Se reutiliza en vez de duplicar.
    nombre_es = decode_shap_feature(shap_feature_name)

    if codigo_ohe is not None:
        # Feature categorica: el codigo decodificado ya es el valor del cliente.
        # No se necesita client_data porque el nombre de la columna OHE ya
        # incluye el valor (p.ej. 'cat__checking_status_A14' -> codigo = 'A14').
        valor_es = decode_feature_value(nombre_columna, codigo_ohe)
    else:
        # Feature numerica: obtener el valor real del cliente y formatearlo.
        valor_crudo = client_data.get(nombre_columna, "")
        valor_es = _format_numeric_value(nombre_columna, valor_crudo)

    return {
        "feature_name":  nombre_es,
        "feature_value": valor_es,
        # CORRECTED: redondeo a 2 decimales en vez de 4 -- shap_value ya
        # esta en puntos de score (rango tipico de decenas), no en
        # probabilidad (rango 0-1); 4 decimales aportaban precision
        # espuria para esta escala.
        "shap_value":    round(float(shap_value), 2),
        # Campos adicionales para trazabilidad interna y el GRADER:
        "feature_raw":   nombre_columna,
        "codigo_ohe":    codigo_ohe,
    }


# ---------------------------------------------------------------------------
# CARGA, AUDITORIA Y PREPROCESAMIENTO
# ---------------------------------------------------------------------------

def load_german_credit(path: str = "data/german-credit-data/german.data") -> pd.DataFrame:
    """
    Carga el dataset original (separado por espacios, sin cabecera) y
    elimina duplicados exactos.
    """
    df = pd.read_csv(path, sep=" ", header=None, names=COLUMN_NAMES)
    df["target"] = df["target"].map({1: 0, 2: 1})

    n_antes = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_eliminadas = n_antes - len(df)
    if n_eliminadas > 0:
        print(f"[load_german_credit] {n_eliminadas} filas duplicadas eliminadas.")

    return df


def audit_data_quality(df: pd.DataFrame) -> dict:
    """Duplicados, nulos y valores fuera de rango plausible."""
    n_duplicados = df.duplicated().sum()

    rangos_plausibles = {
        "age": (18, 100),
        "duration": (1, 120),
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
    """Wrapper sobre dataset_utils.split_data."""
    return _split_data_generic(df, target_col="target", test_size=test_size, seed=seed)


if __name__ == "__main__":
    df = load_german_credit()
    print(f"Filas: {len(df)}, columnas: {len(df.columns)}")
    print(f"Distribucion del target:\n{df['target'].value_counts(normalize=True)}")
    print(f"\nAuditoria de calidad de datos:")
    for k, v in audit_data_quality(df).items():
        print(f"  {k}: {v}")
    print(f"\nVariables sensibles excluidas del modelo: {SENSITIVE_FEATURES_EXCLUDED}")

    # Verificacion del builder de tuplas con un cliente de ejemplo
    cliente_demo = {
        "checking_status": "A11", "duration": 24, "credit_history": "A32",
        "credit_amount": 3500, "age": 34,
    }
    print("\nEjemplo de tuplas para el NARRATOR:")
    # CORRECTED: los valores de ejemplo ahora estan en PUNTOS DE SCORE
    # (los que produciria node_explainability tras la conversion), no en
    # espacio de probabilidad como antes -- consistente con el resto del
    # pipeline. shap_value > 0 -> sube el score.
    casos = [
        ("cat__checking_status_A11", +87.1),
        ("cat__credit_history_A32", -52.0),
        ("num__duration", -107.4),
        ("num__credit_amount", +104.1),
        ("num__age", -32.0),
    ]
    for feat, shap_val in casos:
        t = build_narrator_tuple(feat, cliente_demo, shap_val)
        print(f"  ({t['feature_name']}, {t['feature_value']}, {t['shap_value']:+.1f} pts)")
