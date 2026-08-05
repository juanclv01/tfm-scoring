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
    # CORRECTED: se anade 'aprobado' como output de Nodo 1. Necesario para
    # construir las narrativas de ejemplo de Nodo 3 (necesitan saber si la
    # solicitud fue aprobada o rechazada, no solo el score numerico). La
    # decision usa el umbral optimo de coste (compute_decision_threshold.py),
    # NO un corte fijo en score=500/proba=0.5 -- ver justificacion en
    # evaluate.py (docstring de find_cost_optimal_threshold).
    aprobado: Optional[bool]               # output Nodo 1
    # CORRECTED: se anade 'nivel_riesgo' ("bajo"/"moderado"/"alto"), la
    # etiqueta cualitativa que necesita el NARRATOR para las narrativas
    # de ejemplo -- calculada en codigo (scoring.clasificar_riesgo), no
    # por el LLM, para que el prompt del NARRATOR nunca contenga los
    # umbrales numericos (ni se filtren al cliente, ni queden hardcodeados
    # frente a un modelo que puede reentrenarse). Ver docstring de
    # scoring.clasificar_riesgo() para la justificacion completa.
    nivel_riesgo: Optional[str]            # output Nodo 1
    shap_values: Optional[list]            # output Nodo 2
    base_value: Optional[float]            # output Nodo 2
    feature_names: Optional[list]          # output Nodo 2 (nombres transformados, p.ej. 'cat__checking_status_A14' -- uno por columna dummy, SIN agregar)
    # CORRECTED: top_features ya NO son "nombres crudos" ni columnas dummy
    # individuales -- son las 5 VARIABLES ORIGINALES (categoricas ya
    # agregadas sumando sus dummies, numericas sin cambios) con mayor
    # |SHAP| agregado, decodificadas a espanol y con feature_value tomado
    # siempre del valor real del cliente (ver
    # data_loader.aggregate_shap_by_feature para el bug de mezclar
    # categorias mutuamente excluyentes que esto corrige).
    top_features: Optional[list]           # output Nodo 2 (top-5 variables por |SHAP| agregado, ya en formato NARRATOR)

    # CORRECTED: 'top_features' (Nodo 2) ya viene completamente decodificado
    # -- node_explainability() llama a data_loader.aggregate_shap_by_feature()
    # + data_loader.build_narrator_tuple() internamente, no solo un nombre
    # crudo. 'top_features_decoded' queda como alias reservado por si el
    # Nodo 3 necesita aplicar una transformacion adicional antes del prompt
    # (p.ej. formatear la cadena final), pero NO requiere volver a llamar a
    # ninguna funcion de decodificacion -- ya no existe 'decode_shap_feature()'
    # en el flujo de build_narrator_tuple (fue reemplazada por
    # aggregate_shap_by_feature(), que opera sobre la variable original, no
    # sobre columnas dummy individuales).
    top_features_decoded: Optional[list]   # output Nodo 3 (o pre-Nodo 3): version legible de top_features
    narrative: Optional[str]               # output Nodo 3: texto regulatorio generado por el LLM

    # NODE_3_ENTRY_POINT: validation_result lo poblara el Nodo 4, no el 3;
    # se reserva aqui para no tener que volver a tocar este archivo dos veces.
    validation_result: Optional[dict]      # output Nodo 4: {"aprobado": bool, "motivo": str}
