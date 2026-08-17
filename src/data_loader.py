"""
Carga y preprocesamiento del German Credit Dataset (UCI).
Fuente: archive.ics.uci.edu/dataset/144/statlog+german+credit+data

NOTA SOBRE VARIABLES SENSIBLES (decision de diseno documentada):
"personal_status" codifica sexo Y estado civil en una unica columna,
y la codificacion de "foreign_worker" tiene un error de documentacion
conocido en la version original de UCI: esta INVERTIDA, de forma que,
tomada literalmente, menos del 5% de los solicitantes serian alemanes --
implausible para un banco regional aleman de los anos 70 (ver Fabris et
al., "Algorithmic Fairness Datasets: the Story so Far", 2022, Data Mining
and Knowledge Discovery, arXiv:2202.01711).
#
# CORRECTED: la cita en comentarios anteriores de este fichero decia
# "Ferrando et al." -- verificado contra la fuente primaria, el autor
# principal es Fabris (A. Fabris, S. Messina, G. Silvello, G. A. Susto).
# Revisar y corregir esta misma cita alli donde se haya usado en la
# memoria del TFM.
#
# El error de foreign_worker se corrige aqui en FEATURE_VALUE_LABELS
# (los codigos A201/A202 se decodifican intercambiados respecto a la
# documentacion oficial de UCI, que esta invertida). Ambas variables
# ("personal_status", "foreign_worker") se EXCLUYEN deliberadamente del
# conjunto de features del modelo, en linea con el Art. 10 del EU AI Act
# -- se mantienen decodificadas solo a efectos de auditoria, nunca se
# exponen al NARRATOR.
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
        "A13": "saldo de 200 DM o más en cuenta corriente, o nómina domiciliada al menos 1 año",
        "A14": "sin cuenta corriente",
    },
    "credit_history": {
        "A30": "sin créditos previos o todos pagados puntualmente en otras entidades",
        "A31": "todos los créditos en este banco pagados puntualmente",
        "A32": "créditos existentes pagados puntualmente hasta la fecha",
        "A33": "retraso en pagos registrado en el pasado",
        "A34": "cuenta crítica o créditos existentes en otras entidades",
    },
    "purpose": {
        "A40": "compra de coche nuevo",
        "A41": "compra de coche usado",
        "A42": "compra de mobiliario o equipamiento del hogar",
        "A43": "compra de televisión o radio",
        "A44": "compra de electrodomésticos",
        "A45": "reparaciones del hogar",
        "A46": "educación o formación",
        "A47": "vacaciones",
        "A48": "reciclaje o formación profesional",
        "A49": "negocio o actividad empresarial",
        "A410": "otros propósitos",
    },
    "savings_status": {
        "A61": "ahorros inferiores a 100 DM",
        "A62": "ahorros entre 100 y 500 DM",
        "A63": "ahorros entre 500 y 1000 DM",
        "A64": "ahorros de 1000 DM o más",
        "A65": "sin cuenta de ahorros o importe desconocido",
    },
    "employment": {
        "A71": "desempleado",
        "A72": "empleado desde hace menos de 1 año",
        "A73": "empleado entre 1 y 4 años",
        "A74": "empleado entre 4 y 7 años",
        "A75": "empleado desde hace 7 años o más",
    },
    "personal_status": {  # EXCLUIDA DEL MODELO -- solo auditoría
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
        "A174": "directivo, autónomo o empleado altamente cualificado",
    },
    "own_telephone": {
        "A191": "sin teléfono registrado a su nombre",
        "A192": "teléfono registrado a nombre del solicitante",
    },
    # CORRECTED: codigos A201/A202 INTERCAMBIADOS respecto a la
    # documentacion oficial de UCI. Fabris et al. (2022) confirman que la
    # codificacion original esta invertida: tomada literalmente, menos
    # del 5% de los solicitantes serian alemanes -- implausible para un
    # banco regional aleman de los 70. La version aqui ya refleja el
    # significado real, no el codigo oficial (erroneo) de UCI.
    "foreign_worker": {  # EXCLUIDA DEL MODELO -- solo auditoría
        "A201": "no es trabajador extranjero",
        "A202": "trabajador extranjero",
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
    "purpose":              "Propósito del crédito",
    "savings_status":       "Estado de la cuenta de ahorros",
    "employment":           "Situación laboral",
    "other_parties":        "Avalistas o codeudores",
    "property_magnitude":   "Propiedades del solicitante",
    "other_payment_plans":  "Otros planes de pago activos",
    "housing":              "Tipo de vivienda",
    "job":                  "Categoría profesional",
    "own_telephone":        "Teléfono registrado",
    # Numericas
    "duration":             "Duración del préstamo (meses)",
    "credit_amount":        "Importe del crédito (DM)",
    "installment_rate":     "Cuota como porcentaje de ingresos",
    "residence_since":      "Años en la residencia actual",
    "age":                  "Edad del solicitante (años)",
    "existing_credits":     "Número de créditos activos en este banco",
    "num_dependents":       "Número de personas a cargo",
    # Excluidas del modelo -- se incluyen solo para trazabilidad
    "personal_status":      "Estado civil y sexo (EXCLUIDA)",
    "foreign_worker":       "Trabajador extranjero (EXCLUIDA)",
}

# ---------------------------------------------------------------------------
# CORRECTED: resuelta la nota pendiente sobre "installment_rate". Statlog
# UCI documenta el Attribute 8 solo como "(numerical)" sin explicar los
# codigos 1-4 -- la propia UCI reconoce en la version corregida del
# dataset ("South German Credit", Groemping 2019, UCI dataset 573) que la
# codificacion original "sufre errores serios... y no viene con ninguna
# informacion de contexto". La documentacion completa (recuperada de
# CASdatasets, que expone los codigos A81-A84 que la pagina oficial omite)
# confirma que Attribute 8 NO es un porcentaje continuo -- es un TRAMO
# ordinal, con la convencion (contraintuitiva: a mayor codigo, MENOR
# tramo) A81 > 35%, A82 en [25,35), A83 en [20,25), A84 < 20%. El template
# anterior ("{value}% de los ingresos netos") trataba el codigo como si
# fuera el porcentaje literal (p.ej. codigo=2 -> "2% de los ingresos"),
# una imprecision factual real en la narrativa de cara al cliente, no solo
# de estilo -- el codigo 2 corresponde en realidad a 25-35%, no a 2%.
#
# El modelo (Nodo 1) sigue tratando installment_rate como passthrough
# numerico -- eso no cambia, es una decision de modelado razonable para
# una variable ordinal. Lo que cambia es UNICAMENTE como se decodifica
# para el NARRATOR (Nodo 2/3): mediante un diccionario de tramos, igual
# que una variable categorica, en vez de un template de "{value}%".
INSTALLMENT_RATE_LABELS = {
    1: "más del 35% de los ingresos netos",
    2: "entre el 25% y el 35% de los ingresos netos",
    3: "entre el 20% y el 25% de los ingresos netos",
    4: "menos del 20% de los ingresos netos",
}
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# CORRECTED: "num_dependents" (Attribute 18) tiene el MISMO problema que
# installment_rate -- Statlog UCI lo documenta solo como "(numerical)",
# pero en los datos solo toma los valores {1, 2}, rango demasiado estrecho
# para un conteo literal de personas a cargo entre 1000 solicitantes
# distintos. A diferencia de "residence_since"/"existing_credits" (ver
# nota mas abajo), aqui SI hay una fuente fiable que confirma el tramo:
# un trabajo que reconstruye explicitamente la codificacion corregida de
# South German Credit (Groemping, 2019) documenta esta variable como
# "3 o mas" vs. "0 a 2" -- es decir, el codigo 1 = "0 a 2 personas a
# cargo", el codigo 2 = "3 o mas". El template numerico anterior
# ("{value} persona(s) a cargo") trataba el codigo como un conteo
# literal (1 persona, 2 personas), ocultando que en realidad el codigo 2
# no significa "exactamente 2" sino "3 o mas".
NUM_DEPENDENTS_LABELS = {
    1: "entre 0 y 2 personas a cargo",
    2: "3 o más personas a cargo",
}
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# FORMATO DE VALORES NUMERICOS PARA EL NARRATOR
# Permite presentar los valores numericos con contexto y unidades,
# no como cifras aisladas. El template usa {value} como placeholder.
#
# RESUELTO: "residence_since" y "existing_credits" se verificaron contra
# multiples fuentes independientes (no solo CASdatasets, que ya se
# demostro poco fiable para esta clase de codigos -- reutilizaba
# erroneamente los codigos de "employment" para "residence_since").
# Dos fuentes independientes describen "residence_since" explicitamente
# como anos literales ("Present residence since X years"), y otras dos
# describen "existing_credits" como un conteo entero literal
# ("Number of existing credits at this bank", dominio entero >= 0), sin
# mencionar tramos. Con evidencia externa consistente en ambos casos, se
# mantienen como valores NUMERICOS CONTINUOS (sin diccionario de tramos),
# a diferencia de installment_rate y num_dependents, para los que SI
# existe una fuente que documenta explicitamente su naturaleza de tramo.
NUMERIC_FEATURE_FORMAT = {
    "duration":          ("{value} mes", "{value} meses"),
    "credit_amount":     ("{value} DM", "{value} DM"),
    "residence_since":   ("{value} año", "{value} años"),
    "age":               ("{value} año", "{value} años"),
    "existing_credits":  ("{value} crédito", "{value} créditos"),
}


def _format_numeric_value(feature_name: str, value) -> str:
    """
    Formatea un valor numerico aplicando el template de unidades -- o,
    para variables ordinales por tramos (installment_rate,
    num_dependents), el diccionario de tramos correspondiente en vez de
    tratar el codigo como una cifra continua.

    # CORRECTED: bug de concordancia de numero detectado al enumerar los
    # valores posibles para redactar narrativas -- el template anterior
    # ("{value} anos"/"{value} credito(s)") producia "1 anos" (incorrecto
    # en espanol) para cualquier cliente con residence_since=1, y el
    # parche "(s)" en credito(s) no es una forma real del idioma. Ahora
    # NUMERIC_FEATURE_FORMAT guarda un par (singular, plural) por feature
    # y se elige segun el valor real -- singular solo para 1 exacto
    # (no para 1.0 vs 1.5, aunque estas features son siempre enteras en
    # la practica).
    """
    variables_por_tramo = {
        "installment_rate": INSTALLMENT_RATE_LABELS,
        "num_dependents": NUM_DEPENDENTS_LABELS,
    }
    if feature_name in variables_por_tramo:
        try:
            return variables_por_tramo[feature_name].get(int(value), str(value))
        except (ValueError, TypeError):
            return str(value)

    if feature_name not in NUMERIC_FEATURE_FORMAT:
        return str(value)

    template_singular, template_plural = NUMERIC_FEATURE_FORMAT[feature_name]
    try:
        valor_num = int(value) if float(value) == int(float(value)) else round(float(value), 2)
    except (ValueError, TypeError):
        return str(value)

    template = template_singular if valor_num == 1 else template_plural
    return template.format(value=valor_num)


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


def aggregate_shap_by_feature(feature_names: list, shap_values) -> dict:
    """
    Agrupa los valores SHAP de las columnas dummy (OneHotEncoder) que
    pertenecen a la misma variable categorica original, sumando sus
    contribuciones en una sola cifra por variable. Las features
    numericas (una unica columna transformada por variable) pasan sin
    cambios, con su propia clave.

    # CORRECTED (bug detectado por revision manual de narrativas de
    # ejemplo, no por test automatico): build_narrator_tuple() decodificaba
    # antes 'feature_value' a partir del codigo incrustado en el nombre de
    # la columna dummy que aparecia en el top-5 (p.ej.
    # 'cat__checking_status_A14' -> 'sin cuenta corriente'), SIN comprobar
    # si esa dummy concreta valia 1 (categoria real del cliente) o 0
    # (cliente NO esta en esa categoria). Con OneHotEncoder solo una
    # dummy por variable categorica vale 1 para cada cliente; SHAP puede
    # asignar contribucion no nula a dummies en 0 (el modelo usa la
    # AUSENCIA de esa categoria, no su presencia). Resultado observado:
    # dos dummies de la MISMA variable (checking_status: A14 y A11)
    # aparecian ambas en el top-5 de un mismo cliente, decodificadas como
    # si el cliente tuviera simultaneamente dos estados de cuenta
    # corriente mutuamente excluyentes -- una contradiccion factual
    # imposible, no un comportamiento legitimo de SHAP con OHE.
    #
    # La correccion suma TODAS las dummies de una misma variable
    # categorica en una sola cifra antes de seleccionar el top-5.
    # Matematicamente no altera nada: sum(shap_values) se preserva
    # exactamente igual (es una reagrupacion de la suma, no un cambio de
    # valores), asi que la propiedad de local accuracy verificada sobre
    # el array COMPLETO en test_graph.py no se ve afectada -- este
    # agregado se usa UNICAMENTE para construir top_features (la vista
    # narrativa), nunca para state["shap_values"]/state["base_value"].
    """
    agregados: dict = {}
    for nombre_transformado, valor in zip(feature_names, shap_values):
        nombre_columna, _ = parse_onehot_feature_name(nombre_transformado)
        agregados[nombre_columna] = agregados.get(nombre_columna, 0.0) + float(valor)
    return agregados


def build_narrator_tuple(
    nombre_columna: str,
    client_data: dict,
    shap_value: float,
) -> dict:
    """
    Construye la tupla completa que necesita el NARRATOR segun el formato
    de Explingo (Zytek et al., 2024):
        (feature_name, feature_value, SHAP contribution)

    donde feature_name y feature_value estan en espanol y son directamente
    legibles por el cliente final sin necesidad de conocer los codigos de UCI.

    # CORRECTED: el primer argumento paso de ser 'shap_feature_name' (el
    # nombre de una columna dummy transformada, p.ej.
    # 'cat__checking_status_A14') a ser 'nombre_columna', el nombre de la
    # VARIABLE ORIGINAL (p.ej. 'checking_status'), ya agregada por
    # aggregate_shap_by_feature(). Esto es lo que permite que
    # feature_value se decodifique SIEMPRE a partir del valor real del
    # cliente en client_data (nunca del codigo de una dummy concreta que
    # podria valer 0 para este cliente) -- ver docstring de
    # aggregate_shap_by_feature() para el bug que esto corrige.

    Para features categoricas: decodifica el valor REAL del cliente
    (client_data[nombre_columna]) a su descripcion en espanol -- nunca el
    codigo de una dummy OHE especifica, que podria no ser la categoria
    real del cliente.
    Para features numericas: formatea el valor numerico con sus unidades.

    Args:
        nombre_columna: nombre de la variable ORIGINAL, ya agregada
                        (p.ej. 'checking_status', 'age') -- no el nombre
                        de una columna dummy transformada.
        client_data:    diccionario con los datos crudos del cliente
                        (las mismas claves que COLUMN_NAMES, sin 'target').
        shap_value:     contribucion YA AGREGADA (suma de todas las dummies
                        de esta variable si es categorica) y YA CONVERTIDA
                        a PUNTOS DE SCORE por node_explainability (Nodo 2).
                        Convencion: shap_value > 0 -> SUBE el score;
                        shap_value < 0 -> BAJA el score.

    Returns:
        Diccionario con las tres claves del formato Explingo:
            feature_name    (str, en espanol)
            feature_value   (str, legible, en espanol, SIEMPRE el valor
                            real del cliente)
            shap_value      (float, PUNTOS DE SCORE, ya agregado por
                            variable si es categorica)
    """
    nombre_es = FEATURE_NAME_LABELS_ES.get(nombre_columna, nombre_columna)

    if nombre_columna in CATEGORICAL_FEATURES:
        codigo_real = client_data.get(nombre_columna, "")
        valor_es = decode_feature_value(nombre_columna, codigo_real)
        codigo_ohe = codigo_real
    else:
        valor_crudo = client_data.get(nombre_columna, "")
        valor_es = _format_numeric_value(nombre_columna, valor_crudo)
        codigo_ohe = None

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
    # CORRECTED: build_narrator_tuple() ahora recibe el nombre de la
    # VARIABLE ORIGINAL, ya agregada por aggregate_shap_by_feature() (no
    # el nombre de una columna dummy individual como 'cat__checking_status_A11').
    # feature_value se decodifica siempre a partir de cliente_demo (el
    # valor real del cliente), nunca de un codigo embebido en el nombre.
    # Valores de ejemplo en PUNTOS DE SCORE (los que produciria
    # node_explainability tras la conversion): shap_value > 0 -> sube el score.
    casos = [
        ("checking_status", +87.1),
        ("credit_history", -52.0),
        ("duration", -107.4),
        ("credit_amount", +104.1),
        ("age", -32.0),
    ]
    for nombre_columna, shap_val in casos:
        t = build_narrator_tuple(nombre_columna, cliente_demo, shap_val)
        print(f"  ({t['feature_name']}, {t['feature_value']}, {t['shap_value']:+.1f} pts)")
