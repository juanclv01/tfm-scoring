"""
Ensamblado del grafo LangGraph: Nodo 1 -> Nodo 2.
Los Nodos 3 (LLM narrador) y 4 (validador GDPR) se anadiran como
extension del mismo grafo en la siguiente fase del TFM.
"""
from langgraph.graph import StateGraph, END

from graph_state import PipelineState
from nodes import node_scoring, node_explainability


def build_graph():
    graph = StateGraph(PipelineState)

    graph.add_node("scoring", node_scoring)
    graph.add_node("explainability", node_explainability)

    graph.set_entry_point("scoring")
    graph.add_edge("scoring", "explainability")
    graph.add_edge("explainability", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_graph()

    # personal_status y foreign_worker se incluyen aqui porque un cliente
    # real trae estos campos en su solicitud -- pero el ColumnTransformer
    # del modelo (ver data_loader.CATEGORICAL_FEATURES) no los referencia,
    # por lo que se ignoran automaticamente al construir la matriz de
    # entrada. No es necesario eliminarlos de este diccionario de ejemplo.
    cliente_ejemplo = {
        "checking_status": "A11", "duration": 24, "credit_history": "A32",
        "purpose": "A43", "credit_amount": 3500, "savings_status": "A61",
        "employment": "A73", "installment_rate": 3, "personal_status": "A93",
        "other_parties": "A101", "residence_since": 2,
        "property_magnitude": "A121", "age": 34,
        "other_payment_plans": "A143", "housing": "A152",
        "existing_credits": 1, "job": "A173", "num_dependents": 1,
        "own_telephone": "A192", "foreign_worker": "A201",
    }

    resultado = app.invoke({"client_data": cliente_ejemplo})

    print(f"Score: {resultado['score']} / 1000")
    print(f"Probabilidad de impago: {resultado['proba_default']:.2%}")
    print("Top 5 factores SHAP (nombre crudo -> version legible, Nodo 3):")
    from data_loader import decode_shap_feature
    for f in resultado["top_features"]:
        legible = decode_shap_feature(f["feature"])
        print(f"  {f['feature']} ({legible}): {f['shap_value']:+.4f}")
