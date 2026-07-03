"""
Transforma la probabilidad de impago en el score 0-1000 (output Nodo 1).
Convencion estandar: a mayor score, menor riesgo de impago.
"""


def probability_to_score(
    proba_default: float, min_score: int = 0, max_score: int = 1000
) -> int:
    """
    Escala lineal invertida:
      proba_default = 0.0 -> score = 1000  (riesgo minimo)
      proba_default = 1.0 -> score = 0     (riesgo maximo)
    """
    score = max_score - (proba_default * (max_score - min_score))
    return int(round(score))
