"""
Ensamblado del grafo LangGraph: Nodo 1 -> Nodo 2 -> Nodo 3 -> Nodo 4,
con la arista condicional de reintento del Nodo 4 hacia el Nodo 3 (o
hacia el nodo de revision humana si se agotan los intentos).

# CORRECTED: los Nodos 3 y 4 se anaden ahora como nodos reales de este
# StateGraph, y el guardarraiz de reintento se implementa mediante
# graph.add_conditional_edges(), no como un bucle Python fuera del
# grafo. Esto es lo que hace de LangGraph algo mas que un envoltorio
# innecesario sobre un pipeline secuencial: sin esta arista condicional,
# los 4 nodos podrian haberse encadenado con funciones normales de
# Python (ver justificacion de por que un pipeline secuencial no basta,
# [REF]).
"""
from langgraph.graph import StateGraph, END

from graph_state import PipelineState
from nodes import node_scoring, node_explainability
from nodes_narrator_grader import (
    node_narrative, node_grader, node_human_review, decide_after_grader,
)


def build_graph():
    graph = StateGraph(PipelineState)

    graph.add_node("scoring", node_scoring)
    graph.add_node("explainability", node_explainability)
    graph.add_node("narrative", node_narrative)
    graph.add_node("grader", node_grader)
    graph.add_node("human_review", node_human_review)

    graph.set_entry_point("scoring")
    graph.add_edge("scoring", "explainability")
    graph.add_edge("explainability", "narrative")
    graph.add_edge("narrative", "grader")

    # Arista condicional: la unica bifurcacion de todo el grafo. Tras
    # evaluar la narrativa, decide_after_grader() decide si el flujo
    # vuelve al Nodo 3 (reintento), termina (aprobada) o pasa al nodo
    # de revision humana (intentos agotados).
    graph.add_conditional_edges(
        "grader",
        decide_after_grader,
        {
            "retry": "narrative",
            "approved": END,
            "escalate": "human_review",
        },
    )
    graph.add_edge("human_review", END)

    return graph.compile()


if __name__ == "__main__":
    # Ejecucion de humo: un cliente de ejemplo a traves del grafo
    # completo, incluyendo el guardarraiz de reintento si el EVALUADOR
    # rechaza la primera narrativa.
    app = build_graph()

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

    print(f"Score: {resultado['score']} / 1000 | aprobado={resultado['aprobado']} "
          f"| riesgo={resultado['nivel_riesgo']}")
    print(f"Intentos del Narrador: {resultado['intentos']}")
    print(f"Narrativa aprobada por el Evaluador: {resultado['validation_result']['aprobado']}")
    print()
    print(resultado["narrative"])
