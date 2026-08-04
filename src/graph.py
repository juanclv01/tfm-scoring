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

    # personal_status y foreign_worker se incluyen porque un cliente
    # real trae todos los campos en su solicitud -- el ColumnTransformer
    # los ignora automaticamente al no estar en CATEGORICAL_FEATURES.
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

    decision = "APROBADA" if resultado["aprobado"] else "RECHAZADA"
    print(f"Solicitud: {decision}")
    print(f"Score: {resultado['score']} / 1000")
    print(f"Probabilidad de impago: {resultado['proba_default']:.2%}")
    print()
    print("Top 5 factores SHAP en formato NARRATOR (feature_name_es, feature_value_es, shap en pts de score):")
    print()
    for f in resultado["top_features"]:
        signo = "+" if f["shap_value"] > 0 else ""
        print(f"  ({f['feature_name']}, {f['feature_value']}, {signo}{f['shap_value']:.2f} pts)")
