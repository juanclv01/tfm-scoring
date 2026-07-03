from typing import TypedDict, Optional


class PipelineState(TypedDict):
    """
    Estado compartido del grafo. Cada nodo lee y escribe sobre este
    diccionario. Las claves no pobladas aun deben inicializarse a None.
    """
    client_data: dict                      # input: features crudas
    score: Optional[int]                   # output Nodo 1
    proba_default: Optional[float]         # output Nodo 1
    shap_values: Optional[list]            # output Nodo 2
    base_value: Optional[float]            # output Nodo 2
    feature_names: Optional[list]          # output Nodo 2
    top_features: Optional[list]           # output Nodo 2 (top-5 |SHAP|)
