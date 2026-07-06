from typing import TypedDict, Optional


class PipelineState(TypedDict):
    """
    Estado compartido del grafo. Cada nodo lee y escribe sobre este
    diccionario. Las claves no pobladas aun deben inicializarse a None.

    # CORRECTED: se anaden los campos que el Nodo 3 necesitara
    (top_features_decoded, narrative) y que el Nodo 4 necesitara
    (validation_result). No se implementa logica para ellos todavia --
    solo se reserva el contrato de datos para que Node 3/4 puedan
    desarrollarse sin tener que romper PipelineState mas adelante.
    """
    client_data: dict                      # input: features crudas
    score: Optional[int]                   # output Nodo 1
    proba_default: Optional[float]         # output Nodo 1
    shap_values: Optional[list]            # output Nodo 2
    base_value: Optional[float]            # output Nodo 2
    feature_names: Optional[list]          # output Nodo 2 (nombres crudos, p.ej. 'cat__checking_status_A14')
    top_features: Optional[list]           # output Nodo 2 (top-5 |SHAP|, nombres crudos)

    # NODE_3_ENTRY_POINT: top_features_decoded debe poblarse aplicando
    # data_loader.decode_shap_feature() a cada entrada de top_features
    # antes de construir el prompt del narrador.
    top_features_decoded: Optional[list]   # output Nodo 3 (o pre-Nodo 3): version legible de top_features
    narrative: Optional[str]               # output Nodo 3: texto regulatorio generado por el LLM

    # NODE_3_ENTRY_POINT: validation_result lo poblara el Nodo 4, no el 3;
    # se reserva aqui para no tener que volver a tocar este archivo dos veces.
    validation_result: Optional[dict]      # output Nodo 4: {"aprobado": bool, "motivo": str}
