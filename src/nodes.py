"""
Nodos del grafo LangGraph. Cada funcion recibe el estado y devuelve
un diccionario parcial con las claves que actualiza.
"""
import joblib
import shap
import pandas as pd

from graph_state import PipelineState

_model = joblib.load("models/xgb_scoring_pipeline.joblib")
_explainer = None    # inicializacion perezosa (requiere el modelo cargado)
_background = None   # muestra de X_train transformado, para model_output=probability


def node_scoring(state: PipelineState) -> dict:
    """NODO 1 -- XGBoost scoring."""
    df = pd.DataFrame([state["client_data"]])
    proba_default = float(_model.predict_proba(df)[0, 1])
    score = int(round(1000 - proba_default * 1000))
    return {"score": score, "proba_default": proba_default}


def node_explainability(state: PipelineState) -> dict:
    """NODO 2 -- SHAP TreeExplainer sobre el XGBoost del pipeline.

    model_output="probability" es deliberado: garantiza que
    sum(shap_values) + base_value == proba_default (local accuracy
    en espacio de probabilidad, no en log-odds). Requiere el background
    generado previamente por prepare_background.py.
    """
    global _explainer, _background
    df = pd.DataFrame([state["client_data"]])

    preprocessor = _model.named_steps["preprocessor"]
    xgb_model = _model.named_steps["model"]
    X_transformed = preprocessor.transform(df)

    if _explainer is None:
        _background = joblib.load("models/background_sample.joblib")
        _explainer = shap.TreeExplainer(
            xgb_model,
            data=_background,
            feature_perturbation="interventional",
            model_output="probability",
        )

    shap_values = _explainer.shap_values(X_transformed)
    base_value = _explainer.expected_value

    # Con model_output="probability" y un solo modelo binario, expected_value
    # puede devolverse como array de 1 elemento o como escalar segun version.
    if hasattr(base_value, "__len__"):
        base_value = base_value[0]

    feature_names = preprocessor.get_feature_names_out().tolist()
    contributions = list(zip(feature_names, shap_values[0]))
    top_features = sorted(
        contributions, key=lambda x: abs(x[1]), reverse=True
    )[:5]

    return {
        "shap_values": shap_values[0].tolist(),
        "base_value": float(base_value),
        "feature_names": feature_names,
        "top_features": [
            {"feature": f, "shap_value": float(v)} for f, v in top_features
        ],
    }


# NODE_3_ENTRY_POINT
# El Nodo 3 (narrador LLM) se define aqui como una tercera funcion con
# la misma forma que las anteriores: node_narrative(state) -> dict.
# No se implementa en esta pasada -- el objetivo de esta correccion es
# dejar todo lo que el Nodo 3 necesita ya disponible y sin bloqueos:
#
#   from data_loader import decode_shap_feature
#
#   def node_narrative(state: PipelineState) -> dict:
#       top_features_decoded = [
#           {**f, "feature_legible": decode_shap_feature(f["feature"])}
#           for f in state["top_features"]
#       ]
#       # ... construir prompt con state["score"], state["proba_default"]
#       # y top_features_decoded, llamar al LLM, devolver:
#       return {"top_features_decoded": top_features_decoded, "narrative": "..."}
#
# graph.py debera anadir: graph.add_node("narrative", node_narrative) y
# graph.add_edge("explainability", "narrative") en lugar de
# graph.add_edge("explainability", END).
