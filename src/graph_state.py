from typing import TypedDict, Optional


class PipelineState(TypedDict):
    """
    Estado compartido del grafo. Cada nodo lee y escribe sobre este
    diccionario. Las claves no pobladas aun deben inicializarse a None
    (o al valor por defecto que cada nodo aplique via .get()).

    # CORRECTED: se anaden 'intentos' y 'motivo_reintento'. La arista
    # condicional del Nodo 4 (decide_after_grader, en
    # nodes_narrator_grader.py) necesita persistir cuantos intentos
    # lleva ya el Nodo 3 y el motivo del ultimo rechazo ENTRE
    # invocaciones del grafo -- LangGraph no tiene una variable de
    # bucle propia como un 'for' imperativo; el contador de reintentos
    # tiene que vivir en el propio estado compartido para que la arista
    # condicional pueda leerlo en cada vuelta.
    """
    client_data: dict                      # input: features crudas
    score: Optional[int]                   # output Nodo 1
    proba_default: Optional[float]         # output Nodo 1
    aprobado: Optional[bool]               # output Nodo 1 (umbral optimo de coste)
    nivel_riesgo: Optional[str]            # output Nodo 1 ("bajo"/"moderado"/"alto")
    shap_values: Optional[list]            # output Nodo 2 (en puntos de score)
    base_value: Optional[float]            # output Nodo 2 (en puntos de score)
    feature_names: Optional[list]          # output Nodo 2 (sin agregar)
    top_features: Optional[list]           # output Nodo 2 (agregado, decodificado)
    top_features_decoded: Optional[list]   # alias reservado, poblado junto a top_features

    narrative: Optional[str]               # output Nodo 3 (y del nodo de revision humana)
    validation_result: Optional[dict]      # output Nodo 4: {"aprobado": bool, "puntuacion_total": float, "detalle": dict, "motivo": Optional[str]}

    intentos: int                          # cuantas veces se ha ejecutado el Nodo 3 en esta invocacion del grafo
    motivo_reintento: str                  # motivo del ultimo rechazo, pasado al Nodo 3 en el siguiente intento; "" en el primero
