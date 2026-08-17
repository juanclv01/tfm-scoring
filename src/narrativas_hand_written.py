# -*- coding: utf-8 -*-
"""
Las 5 narrativas hand-written (H), ya corregidas, en formato estructurado
listo para construir dspy.Example de few-shot. El ORDEN de esta lista es el
orden de anidamiento H=1 (usa solo la nº1), H=3 (usa 1-3), H=5 (usa 1-5) --
ver nota en experiment_config.CONFIGURACIONES_HB.

context es fijo para las 5 (mismo modelo, misma tarea); se repite aqui de
forma explicita para que cada Example sea autocontenido.
"""

CONTEXT = (
    "The ML model predicts a client's credit score, on a 0-1000 scale where "
    "a HIGHER score means LOWER risk of default."
)

EXPLANATION_FORMAT = (
    "(feature_name, feature_value, shap_value). Each shap_value is expressed "
    "in POINTS of the 0-1000 score (not a probability or a percentage). A "
    "positive shap_value RAISES the client's score (improves their risk "
    "profile); a negative shap_value LOWERS it. These are the top 5 factors "
    "ranked by relevance -- not an exhaustive list of everything that "
    "influenced the decision."
)


def _fmt(tuplas: list) -> str:
    return "\n".join(
        f"({nombre}, {valor}, {shap:+.1f} pts)" for nombre, valor, shap in tuplas
    )


NARRATIVAS_HAND_WRITTEN = [
    {
        "indice_test": 45,
        "aprobado": True,
        "nivel_riesgo": "bajo",
        "score": 812,
        "explanation": _fmt([
            ("Estado de la cuenta corriente", "sin cuenta corriente", 79.8),
            ("Otros planes de pago activos", "sin planes de pago adicionales", 10.4),
            ("Historial crediticio", "cuenta crítica o créditos existentes en otras entidades", 8.9),
            ("Duración del préstamo (meses)", "12 meses", 6.3),
            ("Situación laboral", "empleado desde hace 7 años o más", 3.6),
        ]),
        "narrative": (
            "Su solicitud de préstamo personal ha sido aprobada con una puntuación "
            "crediticia de 812 sobre 1000, lo que la clasifica en la categoría de "
            "riesgo bajo, superando de esta forma nuestros criterios de aceptación.\n\n"
            "Los factores que han favorecido el aumento de su puntuación han sido "
            "principalmente que usted no dispone actualmente de una cuenta corriente "
            "activa (+79.8 pts), que usted no está suscrito a planes de pago "
            "adicionales (+10.4 pts), su historial crediticio, que refleja una cuenta "
            "crítica y/o créditos existentes en otras entidades (+8.9 pts), la "
            "duración del préstamo solicitado, que abarca un periodo de 12 meses "
            "(+6.3 pts), y su antigüedad laboral, de 7 años o más como empleado "
            "(+3.6 pts).\n\n"
            "No hay factores de su perfil que hayan contribuido negativamente de "
            "forma notable sobre su puntuación.\n\n"
            "En virtud del artículo 22 del RGPD, tiene derecho a solicitar revisión "
            "humana de esta decisión, conforme a la normativa aplicable."
        ),
    },
    {
        "indice_test": 107,
        "aprobado": False,
        "nivel_riesgo": "moderado",
        "score": 753,
        "explanation": _fmt([
            ("Duración del préstamo (meses)", "9 meses", 51.3),
            ("Estado de la cuenta corriente", "saldo entre 0 y 200 DM en cuenta corriente", -27.3),
            ("Propiedades del solicitante", "propietario de bienes inmuebles", 15.4),
            ("Importe del crédito (DM)", "2118 DM", 10.0),
            ("Cuota como porcentaje de ingresos", "entre el 25% y el 35% de los ingresos netos", 7.5),
        ]),
        "narrative": (
            "Su solicitud de préstamo personal ha sido rechazada con una puntuación "
            "crediticia de 753 sobre 1000, lo que la clasifica en la categoría de "
            "riesgo moderado, lamentablemente no superando nuestros criterios de "
            "aceptación.\n\n"
            "Los factores que han favorecido el aumento de su puntuación han sido "
            "principalmente la duración del préstamo solicitado, que abarca un "
            "periodo de 9 meses (+51.3 pts), su propiedad de bienes inmuebles "
            "(+15.4 pts), el importe del crédito solicitado, por una suma de 2118 DM "
            "(+10.0 pts), y la cuota del préstamo, que supone entre el 25% y el 35% "
            "de sus ingresos netos (+7.5 pts).\n\n"
            "Por otro lado, el factor principal con mayor impacto negativo sobre su "
            "puntuación consiste en el estado de su cuenta corriente, cuyo saldo se "
            "sitúa entre 0 y 200 DM (-27.3 pts).\n\n"
            "En virtud del artículo 22 del RGPD, tiene derecho a solicitar revisión "
            "humana de esta decisión, conforme a la normativa aplicable."
        ),
    },
    {
        "indice_test": 99,
        "aprobado": False,
        "nivel_riesgo": "moderado",
        "score": 688,
        "explanation": _fmt([
            ("Estado de la cuenta de ahorros", "sin cuenta de ahorros o importe desconocido", 23.6),
            ("Otros planes de pago activos", "planes de pago adicionales en otro banco", -16.6),
            ("Estado de la cuenta corriente", "saldo de 200 DM o más en cuenta corriente, o nómina domiciliada al menos 1 año", -14.4),
            ("Cuota como porcentaje de ingresos", "menos del 20% de los ingresos netos", -6.1),
            ("Importe del crédito (DM)", "1445 DM", 5.5),
        ]),
        "narrative": (
            "Su solicitud de préstamo personal ha sido rechazada con una puntuación "
            "crediticia de 688 sobre 1000, lo que la clasifica en la categoría de "
            "riesgo moderado, lamentablemente no superando nuestros criterios de "
            "aceptación.\n\n"
            "Los factores que han favorecido el aumento de su puntuación han sido "
            "principalmente que usted no dispone actualmente de una cuenta de "
            "ahorros, o su importe es desconocido (+23.6 pts), y el importe del "
            "crédito solicitado, por un valor de 1445 DM (+5.5 pts).\n\n"
            "Por otro lado, los factores principales con mayor impacto negativo sobre "
            "su puntuación consisten en su suscripción a planes de pago adicionales "
            "en otros bancos (-16.6 pts), el estado de su cuenta corriente, con un "
            "saldo de 200 DM o más, o con nómina domiciliada desde hace al menos 1 "
            "año (-14.4 pts), y la cuota del préstamo, que supone menos del 20% de "
            "sus ingresos netos (-6.1 pts).\n\n"
            "En virtud del artículo 22 del RGPD, tiene derecho a solicitar revisión "
            "humana de esta decisión, conforme a la normativa aplicable."
        ),
    },
    {
        "indice_test": 154,
        "aprobado": False,
        "nivel_riesgo": "alto",
        "score": 640,
        "explanation": _fmt([
            ("Estado de la cuenta corriente", "saldo negativo en cuenta corriente", -59.5),
            ("Estado de la cuenta de ahorros", "ahorros inferiores a 100 DM", -11.9),
            ("Importe del crédito (DM)", "2511 DM", 9.8),
            ("Cuota como porcentaje de ingresos", "más del 35% de los ingresos netos", 9.6),
            ("Historial crediticio", "créditos existentes pagados puntualmente hasta la fecha", -6.0),
        ]),
        "narrative": (
            "Su solicitud de préstamo personal ha sido rechazada con una puntuación "
            "crediticia de 640 sobre 1000, lo que la clasifica en la categoría de "
            "riesgo alto, lamentablemente no superando nuestros criterios de "
            "aceptación.\n\n"
            "Los factores que han favorecido el aumento de su puntuación han sido "
            "principalmente el importe del crédito solicitado, por un valor de "
            "2511 DM (+9.8 pts), y la cuota del préstamo, que supone más del 35% de "
            "sus ingresos netos (+9.6 pts).\n\n"
            "Por otro lado, los factores que han reducido su puntuación en mayor "
            "magnitud consisten en el saldo negativo presente en su cuenta corriente "
            "(-59.5 pts), unos ahorros inferiores a 100 DM en su cuenta de ahorros "
            "(-11.9 pts), y su historial crediticio, que refleja créditos existentes "
            "pagados puntualmente hasta la fecha (-6.0 pts).\n\n"
            "En virtud del artículo 22 del RGPD, tiene derecho a solicitar revisión "
            "humana de esta decisión, conforme a la normativa aplicable."
        ),
    },
    {
        "indice_test": 68,
        "aprobado": False,
        "nivel_riesgo": "alto",
        "score": 575,
        "explanation": _fmt([
            ("Importe del crédito (DM)", "18424 DM", -33.3),
            ("Duración del préstamo (meses)", "48 meses", -30.9),
            ("Estado de la cuenta corriente", "saldo entre 0 y 200 DM en cuenta corriente", -30.2),
            ("Estado de la cuenta de ahorros", "ahorros inferiores a 100 DM", -18.9),
            ("Otros planes de pago activos", "planes de pago adicionales en otro banco", -17.3),
        ]),
        "narrative": (
            "Su solicitud de préstamo personal ha sido rechazada con una puntuación "
            "crediticia de 575 sobre 1000, lo que la clasifica en la categoría de "
            "riesgo alto, lamentablemente no superando nuestros criterios de "
            "aceptación.\n\n"
            "Tras el análisis de su perfil, no podemos identificar ningún factor "
            "cuyo efecto haya sido notablemente positivo sobre su puntuación.\n\n"
            "Sin embargo, existen múltiples factores de su perfil que han actuado "
            "considerablemente en la reducción de su puntuación. Entre ellos se "
            "encuentran el importe del crédito solicitado, por un valor de 18424 DM "
            "(-33.3 pts), la duración del préstamo seleccionado, de 48 meses "
            "(-30.9 pts), el estado de su cuenta corriente, con un saldo entre 0 y "
            "200 DM (-30.2 pts), unos ahorros inferiores a 100 DM en su cuenta de "
            "ahorros (-18.9 pts), y la presencia de planes de pago adicionales en "
            "otro banco (-17.3 pts).\n\n"
            "En virtud del artículo 22 del RGPD, tiene derecho a solicitar revisión "
            "humana de esta decisión, conforme a la normativa aplicable."
        ),
    },
]


def obtener_demos_narrator(h: int) -> list:
    """Primeras h narrativas hand-written, en formato dspy.Example, listas
    para asignarse a predict.demos. h debe estar entre 0 y 5."""
    import dspy

    if not (0 <= h <= len(NARRATIVAS_HAND_WRITTEN)):
        raise ValueError(f"h={h} fuera de rango [0, {len(NARRATIVAS_HAND_WRITTEN)}]")

    demos = []
    for item in NARRATIVAS_HAND_WRITTEN[:h]:
        ex = dspy.Example(
            context=CONTEXT,
            decision="APROBADA" if item["aprobado"] else "RECHAZADA",
            risk_level=item["nivel_riesgo"],
            score=item["score"],
            explanation=item["explanation"],
            explanation_format=EXPLANATION_FORMAT,
            narrative=item["narrative"],
        ).with_inputs("context", "decision", "risk_level", "score", "explanation", "explanation_format")
        demos.append(ex)
    return demos
