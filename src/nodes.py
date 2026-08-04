"""
Nodos del grafo LangGraph. Cada funcion recibe el estado y devuelve
un diccionario parcial con las claves que actualiza.

CAMBIO RESPECTO A LA VERSION ANTERIOR:
node_explainability() ahora construye top_features en el formato de
tupla completa que necesita el NARRATOR (Zytek et al., 2024):
    (feature_name_es, feature_value_es, shap_contribution)

donde feature_name y feature_value estan ya en espanol y son
directamente legibles por el cliente final. Para ello llama a
data_loader.build_narrator_tuple() en vez de devolver el nombre
crudo del preprocesador.

# CORRECTED: node_scoring() reimplementaba en linea la misma formula que
# ya existia en scoring.py (probability_to_score), sin llamarla -- logica
# duplicada entre dos ficheros. Ahora importa y usa esa funcion.

# CORRECTED: node_explainability() convertia los shap_values y base_value
# del explainer (espacio de proba_default, P(target=1)="malo") y los
# devolvia tal cual al estado. Con model_output="probability", un
# shap_value positivo sube la probabilidad de impago -> BAJA el score --
# contraintuitivo para quien lea 'top_features' o 'shap_values' esperando
# que positivo signifique "mejora el score". Se aniade una conversion a
# PUNTOS DE SCORE (misma escala 0-1000 que probability_to_score) para que
# la convencion de signo sea la deseada en todo el pipeline: shap_value
# positivo ahora SUBE el score; negativo lo BAJA. La conversion se aplica
# al array COMPLETO (no solo al top-5 narrativo) para que no convivan dos
# convenciones de signo distintas dentro de PipelineState.
"""
import joblib
import shap
import pandas as pd

from graph_state import PipelineState
from data_loader import build_narrator_tuple
from scoring import probability_to_score, SCORE_MAX, SCORE_MIN

_model = joblib.load("models/xgb_scoring_pipeline.joblib")
# CORRECTED: se carga aqui, una sola vez al importar el modulo (mismo
# patron que _model), el umbral de decision optimo de coste calculado
# offline por compute_decision_threshold.py. NO se recalcula en cada
# invocacion del grafo -- requeriria recargar el dataset completo y
# rehacer el split por cada solicitud de un cliente, coste innecesario
# para un valor que no cambia entre solicitudes salvo que se reentrene
# el modelo.
_decision_threshold = joblib.load("models/decision_threshold.joblib")
_explainer = None
_background = None


def node_scoring(state: PipelineState) -> dict:
    """
    NODO 1 -- XGBoost scoring.

    # CORRECTED: se anade 'aprobado' al output. La decision NO usa el
    # umbral neutro proba_default >= 0.5 (score < 500) -- ese umbral no
    # tiene significado de negocio para este problema, ver docstring de
    # evaluate.py. Se usa en su lugar _decision_threshold, el umbral que
    # minimiza el coste esperado segun la matriz 5:1 oficial de UCI
    # (evaluate.find_cost_optimal_threshold(), persistido por
    # compute_decision_threshold.py). Misma convencion de evaluate.py:
    # y_pred=1 ("malo"/rechazado) cuando proba_default >= umbral; por
    # tanto aprobado = proba_default < umbral.
    """
    df = pd.DataFrame([state["client_data"]])
    proba_default = float(_model.predict_proba(df)[0, 1])
    score = probability_to_score(proba_default)
    aprobado = bool(proba_default < _decision_threshold)
    return {"score": score, "proba_default": proba_default, "aprobado": aprobado}


def node_explainability(state: PipelineState) -> dict:
    """
    NODO 2 -- SHAP TreeExplainer.

    model_output='probability' sigue siendo la configuracion correcta del
    explainer: es lo que garantiza matematicamente la propiedad de local
    accuracy (Lundberg & Lee, 2017) en el momento en que SHAP calcula los
    valores. Lo que cambia es el POST-PROCESADO inmediatamente despues:
    tanto shap_values como base_value se convierten de espacio de
    proba_default a PUNTOS DE SCORE antes de guardarse en el estado o de
    construir top_features, mediante la misma transformacion lineal
    invertida que usa scoring.probability_to_score() mas arriba, pero sin
    redondear a entero (aqui se necesita precision continua; el redondeo
    a int solo ocurre para el score final del Nodo 1).

    Convencion de signo resultante (deliberada, no la que devuelve SHAP
    por defecto): shap_value > 0 -> SUBE el score; shap_value < 0 -> BAJA
    el score. Se aplica al array COMPLETO, no solo al top-5, para que
    'shap_values' (usado por el test de local accuracy) y 'top_features'
    (usado por el NARRATOR) compartan siempre la misma convencion.

    Local accuracy en la nueva escala:
        sum(shap_values) + base_value == score_continuo
        donde score_continuo = 1000 * (1 - proba_default)
        (verificado en tests/test_graph.py contra proba_default, no
        contra el 'score' entero de node_scoring, que ya lleva redondeo).

    top_features contiene el formato completo para el NARRATOR:
        [
            {
                "feature_name":  "Estado de la cuenta corriente",
                "feature_value": "saldo negativo en cuenta corriente",
                "shap_value":    +87.1,   # puntos de score, no probabilidad
                "feature_raw":   "checking_status",   # para trazabilidad
                "codigo_ohe":    "A11",                # None para numericas
            },
            ...
        ]

    Los campos 'feature_raw' y 'codigo_ohe' no van al prompt del NARRATOR
    pero son utiles para el GRADER (puede verificar fidelidad SHAP
    comparando 'feature_name'/'feature_value' frente a los valores reales).
    """
    global _explainer, _background
    df = pd.DataFrame([state["client_data"]])

    preprocessor = _model.named_steps["preprocessor"]
    xgb_model    = _model.named_steps["model"]
    X_transformed = preprocessor.transform(df)

    if _explainer is None:
        _background = joblib.load("models/background_sample.joblib")
        _explainer  = shap.TreeExplainer(
            xgb_model,
            data=_background,
            feature_perturbation="interventional",
            model_output="probability",
        )

    shap_values_proba = _explainer.shap_values(X_transformed)[0]
    base_value_proba  = _explainer.expected_value
    if hasattr(base_value_proba, "__len__"):
        base_value_proba = base_value_proba[0]

    # Conversion a espacio de score: misma escala lineal invertida que
    # probability_to_score(), aplicada aqui sin redondeo a entero.
    # shap_value > 0 (antes: sube proba_default) -> ahora SUBE el score.
    shap_values_score = -shap_values_proba * (SCORE_MAX - SCORE_MIN)
    base_value_score  = SCORE_MAX - (base_value_proba * (SCORE_MAX - SCORE_MIN))

    feature_names = preprocessor.get_feature_names_out().tolist()

    # Ordenar por |SHAP| descendente y quedarse con los top 5. El orden no
    # cambia respecto a hacerlo en espacio de probabilidad (multiplicar
    # por una constante negativa invierte el signo, no la magnitud
    # relativa), pero se ordena ya en espacio de score para evitar
    # cualquier ambiguedad sobre que array se esta ordenando.
    contributions = list(zip(feature_names, shap_values_score))
    top_5 = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)[:5]

    # Construir las tuplas completas (feature_name_es, feature_value_es, shap)
    # usando build_narrator_tuple de data_loader.py. shap_val ya esta en
    # puntos de score en este punto.
    # client_data se pasa para que las features numericas puedan recuperar
    # el valor real del cliente (p.ej. age=34 -> "34 anos").
    top_features_decoded = [
        build_narrator_tuple(feat_name, state["client_data"], shap_val)
        for feat_name, shap_val in top_5
    ]

    return {
        "shap_values":   shap_values_score.tolist(),
        "base_value":    float(base_value_score),
        "feature_names": feature_names,
        "top_features":  top_features_decoded,
    }


# NODE_3_ENTRY_POINT
# El prompt del NARRATOR debe construirse asi, basandose en top_features:
#
# explanation_lines = [
#     f"({f['feature_name']}, {f['feature_value']}, {f['shap_value']:+.1f} pts)"
#     for f in state["top_features"]
# ]
# explanation_str = "\n".join(explanation_lines)
#
# Convencion de signo (tras la conversion a espacio de score en Nodo 2):
# shap_value > 0 -> SUBE el score; shap_value < 0 -> BAJA el score. Ya no
# requiere ninguna instruccion especial en el prompt para evitar que el
# LLM la invierta -- coincide con la lectura intuitiva "positivo = mejora".
#
# state["aprobado"] (Nodo 1, umbral optimo de coste) ya esta disponible
# para construir la narrativa -- necesario para que el NARRATOR pueda
# comunicar el resultado de la decision, no solo el score numerico
# (imprescindible tambien para generate_shap_examples.py, que reutiliza
# node_scoring()/node_explainability() para las narrativas de ejemplo).
#
# Las claves feature_raw y codigo_ohe NO deben incluirse en el prompt
# del NARRATOR (son informacion tecnica interna); SÍ pueden incluirse
# en el prompt del GRADER para que este verifique fidelidad.
#
# Estructura de node_narrative(state) -> dict:
#   - Lee: state["top_features"], state["score"], state["proba_default"]
#   - Escribe: state["top_features_decoded"] (ya esta en top_features),
#              state["narrative"]
#
# graph.py debera anadir:
#   graph.add_node("narrative", node_narrative)
#   graph.add_edge("explainability", "narrative")
#   (en lugar de graph.add_edge("explainability", END))
