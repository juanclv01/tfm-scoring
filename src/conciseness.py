r"""
Dimension CONCISENESS del GRADER (Zytek et al., 2024) -- calculada de forma
deterministica, sin llamada a LLM, siguiendo la ecuacion:

           / 0                          si L >= 2 * F * L_max
grade  =  | 4 * (2 - L / (F * L_max))   si F*L_max < L < 2*F*L_max
           \ 4                          si L <= F * L_max

donde:
    L      = numero de palabras de la narrativa
    F      = numero de features en la explicacion (5, tamano fijo del top-5)
    L_max  = hiperparametro: numero maximo "ideal" de palabras por feature

Nota de notacion: el paper llama "FL_max" al producto F * L_max (el limite
ideal total de palabras para F features), no una variable independiente.
Se implementa aqui como tal para evitar ambiguedad.

No requiere GRADER LLM: se ejecuta en node_grader() como cualquier otra
funcion determinista del pipeline (igual que la binarizacion de completeness
para el requisito (c) del GDPR).
"""

L_MAX_DEFAULT = 30  # palabras "ideales" por feature -- calibrado por Juan
                     # contra las 5 narrativas H (ver experiment_config.CONCISENESS_L_MAX,
                     # que es el valor que realmente usa metrics.py en el pipeline).
                     # Este default solo se usaria si se llama a conciseness_score()
                     # sin pasar l_max explicitamente (p.ej. pruebas sueltas).


def conciseness_score(narrative: str, num_features: int, l_max: int = L_MAX_DEFAULT) -> float:
    """
    Calcula la puntuacion de conciseness (0-4, continua) de una narrativa.

    Args:
        narrative:    texto de la narrativa generada por el NARRATOR.
        num_features: numero de features en la explicacion (F). En este
                      pipeline es siempre 5 (tamano fijo del top-5), pero se
                      deja como parametro explicito en vez de hardcodear 5,
                      por si en el futuro el top-N deja de ser fijo.
        l_max:        palabras ideales por feature (hiperparametro).

    Returns:
        Puntuacion float en [0, 4].
    """
    if num_features <= 0:
        raise ValueError("num_features debe ser > 0")
    if l_max <= 0:
        raise ValueError("l_max debe ser > 0")

    L = len(narrative.split())
    FL_max = num_features * l_max

    if L >= 2 * FL_max:
        return 0.0
    if L <= FL_max:
        return 4.0
    return 4 * (2 - L / FL_max)


if __name__ == "__main__":
    # Ejemplo con la Instancia 1 corregida (5 features, L_max=15 -> FL_max=75)
    ejemplo = (
        "Su solicitud de prestamo personal ha sido aprobada con una puntuacion "
        "crediticia de 812 sobre 1000, lo que la clasifica en la categoria de "
        "riesgo bajo, superando de esta forma nuestros criterios de aceptacion."
    )
    print(conciseness_score(ejemplo, num_features=5))
