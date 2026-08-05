"""
Transforma la probabilidad de impago en el score 0-1000 (output Nodo 1).
Convencion estandar: a mayor score, menor riesgo de impago.
"""

# CORRECTED: se exponen como constantes de modulo (antes solo existian
# como valores por defecto de los parametros de probability_to_score) para
# que nodes.py pueda reutilizar el mismo rango al convertir los valores
# SHAP de espacio de probabilidad a espacio de score (Nodo 2), sin
# duplicar el numero magico 1000 en dos ficheros distintos.
SCORE_MIN = 0
SCORE_MAX = 1000


def probability_to_score(
    proba_default: float, min_score: int = SCORE_MIN, max_score: int = SCORE_MAX
) -> int:
    """
    Escala lineal invertida:
      proba_default = 0.0 -> score = 1000  (riesgo minimo)
      proba_default = 1.0 -> score = 0     (riesgo maximo)
    """
    score = max_score - (proba_default * (max_score - min_score))
    return int(round(score))


def clasificar_riesgo(
    proba_default: float,
    umbral_bajo_moderado: float,
    umbral_moderado_alto: float,
) -> str:
    """
    Clasifica proba_default en una banda cualitativa ("bajo"/"moderado"/
    "alto") usando dos umbrales calculados empiricamente sobre el test set
    del modelo actual (ver compute_decision_threshold.py) -- nunca cifras
    fijas en el codigo.

    DECISION DE ARQUITECTURA (importante, no cambiar sin releer esto):
    esta funcion es la UNICA responsable de convertir un numero en una
    etiqueta de riesgo. El NARRATOR (Nodo 3) recibe solo el resultado
    ("bajo"/"moderado"/"alto"), nunca proba_default ni los umbrales --
    por dos motivos a la vez:
      1. Los umbrales se recalculan cada vez que se reentrena el modelo
         (compute_decision_threshold.py); si el LLM tuviera que aplicar el
         corte el mismo, el prompt tendria que hardcodear un numero que
         quedaria obsoleto en el siguiente entrenamiento.
      2. Los umbrales de negocio (cuanto riesgo se tolera) no deben
         aparecer en la narrativa de cara al cliente bajo ningun concepto.
    Al resolver la clasificacion aqui, en codigo determinista, ambos
    problemas desaparecen a la vez: el LLM nunca ve un numero que
    clasificar, solo una etiqueta ya decidida.

    Intervalos semiabiertos, sin huecos ni solapes en los bordes,
    consistente con la desigualdad estricta que ya usa node_scoring()
    para 'aprobado' (proba_default < umbral_bajo_moderado):
        bajo:      [0, umbral_bajo_moderado)
        moderado:  [umbral_bajo_moderado, umbral_moderado_alto)
        alto:      [umbral_moderado_alto, 1]

    Con el umbral_bajo_moderado = umbral de aprobacion (uso previsto en
    node_scoring()), 'riesgo bajo' coincide exactamente con 'aprobado' --
    es una consecuencia deliberada del diseño, no una coincidencia:
    'moderado' y 'alto' son, ambas, siempre rechazo; solo aportan matiz
    sobre CUANTO se aleja el cliente del umbral de aprobacion.
    """
    if not (0.0 <= umbral_bajo_moderado <= umbral_moderado_alto <= 1.0):
        raise ValueError(
            f"Umbrales invalidos: se espera "
            f"0 <= umbral_bajo_moderado ({umbral_bajo_moderado}) "
            f"<= umbral_moderado_alto ({umbral_moderado_alto}) <= 1"
        )

    if proba_default < umbral_bajo_moderado:
        return "bajo"
    elif proba_default < umbral_moderado_alto:
        return "moderado"
    else:
        return "alto"
