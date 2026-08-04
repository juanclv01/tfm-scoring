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
