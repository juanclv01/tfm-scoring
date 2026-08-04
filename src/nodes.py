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
"""
import joblib
import shap
import pandas as pd

from graph_state import PipelineState
from data_loader import build_narrator_tuple
from scoring import probability_to_score

_model = joblib.load("models/xgb_scoring_pipeline.joblib")
_explainer = None
_background = None


def node_scoring(state: PipelineState) -> dict:
    """NODO 1 -- XGBoost scoring."""
    df = pd.DataFrame([state["client_data"]])
    proba_default = float(_model.predict_proba(df)[0, 1])
    score = probability_to_score(proba_default)
    return {"score": score, "proba_default": proba_default}


def node_explainability(state: PipelineState) -> dict:
    """
    NODO 2 -- SHAP TreeExplainer.

    model_output='probability' garantiza que:
        sum(shap_values) + base_value == proba_default
    (propiedad de local accuracy, verificada en tests/test_graph.py).

    top_features ahora contiene el formato completo para el NARRATOR:
        [
            {
                "feature_name":  "Estado de la cuenta corriente",
                "feature_value": "saldo negativo en cuenta corriente",
                "shap_value":    +0.0871,
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

    shap_values = _explainer.shap_values(X_transformed)
    base_value  = _explainer.expected_value
    if hasattr(base_value, "__len__"):
        base_value = base_value[0]

    feature_names = preprocessor.get_feature_names_out().tolist()

    # Ordenar por |SHAP| descendente y quedarse con los top 5
    contributions = list(zip(feature_names, shap_values[0]))
    top_5 = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)[:5]

    # Construir las tuplas completas (feature_name_es, feature_value_es, shap)
    # usando build_narrator_tuple de data_loader.py.
    # client_data se pasa para que las features numericas puedan recuperar
    # el valor real del cliente (p.ej. age=34 -> "34 anos").
    top_features_decoded = [
        build_narrator_tuple(feat_name, state["client_data"], shap_val)
        for feat_name, shap_val in top_5
    ]

    return {
        "shap_values":   shap_values[0].tolist(),
        "base_value":    float(base_value),
        "feature_names": feature_names,
        "top_features":  top_features_decoded,
    }


# NODE_3_ENTRY_POINT
# El prompt del NARRATOR debe construirse asi, basandose en top_features:
#
# explanation_lines = [
#     f"({f['feature_name']}, {f['feature_value']}, {f['shap_value']:+.4f})"
#     for f in state["top_features"]
# ]
# explanation_str = "\n".join(explanation_lines)
#
# Convencion de signo (confirmada, no requiere transformacion adicional):
# shap_value > 0 -> sube la probabilidad de impago -> BAJA el score;
# shap_value < 0 -> SUBE el score. El prompt del NARRATOR debe instruir
# explicitamente esta direccion para que la narrativa generada no la
# invierta por generalizacion ingenua de "positivo = bueno".
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
