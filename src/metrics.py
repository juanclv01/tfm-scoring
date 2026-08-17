"""
Calcula las 5 dimensiones del GRADER para una narrativa concreta:
accuracy, completeness, fluency (LLM), conciseness (determinista), y GDPR
(combinacion de LLM(a+b) + completeness binarizada, arquitectura ya
bloqueada -- ver contexto_nodo2.txt Seccion 11).

IMPORTANTE: completeness debe calcularse ANTES que GDPR (dependencia
explicita), tal como se establecio en la conversacion de diseno.
"""
import dspy

from conciseness import conciseness_score
from experiment_config import GRADER_SECONDS_BETWEEN_CALLS, CONCISENESS_L_MAX
from dspy_modules import (
    AccuracySignature, CompletenessSignature, FluencySignature, GdprSignature,
    llamar_con_pausa,
)

_accuracy_predict = dspy.Predict(AccuracySignature)
_completeness_predict = dspy.Predict(CompletenessSignature)
_fluency_predict = dspy.Predict(FluencySignature)
_gdpr_predict = dspy.Predict(GdprSignature)


def gdpr_final_score(completeness_score: int, llm_ab_score: int) -> int:
    """
    completeness_score: 0, 1 o 2 -- salida de CompletenessSignature
    llm_ab_score: 0, 1 o 2 -- salida de GdprSignature (a+b)
    Devuelve 0-3: 0=ninguno, 1=uno de (a,b,c), 2=dos de (a,b,c), 3=los tres.
    """
    c = 1 if completeness_score >= 1 else 0
    return llm_ab_score + c


def evaluar_narrativa(
    narrative: str,
    ground_truth: str,
    explanation: str,
    explanation_format: str,
    exemplars: str,
    num_features: int,
    grader_lm: dspy.LM,
) -> dict:
    """
    Evalua una narrativa en las 5 dimensiones. Devuelve un dict:
        {"accuracy": int, "completeness": int, "fluency": int,
         "conciseness": float, "gdpr": int,
         "justificaciones": {"accuracy": str, "completeness": str,
                              "fluency": str, "gdpr": str}}

    Las justificaciones son SOLO para revision del investigador -- nunca
    deben mostrarse al cliente ni mezclarse con el campo 'narrative'. Se
    devuelven agrupadas bajo su propia clave para que sea imposible
    incluirlas por error junto al resto de scores en ningun renderizado
    de cara al cliente (p.ej. build_results_table.py no las toca porque
    no forman parte de DIMENSIONES).

    exemplars: texto de las narrativas hand-written usadas como estilo de
               referencia para fluency (puede ser cadena vacia si H=0 en
               esa configuracion -- ver FluencySignature).
    """
    acc = llamar_con_pausa(
        _accuracy_predict, grader_lm, GRADER_SECONDS_BETWEEN_CALLS,
        ground_truth=ground_truth, explanation=explanation,
        explanation_format=explanation_format, narrative=narrative,
    )
    accuracy = int(acc.assessment)

    comp = llamar_con_pausa(
        _completeness_predict, grader_lm, GRADER_SECONDS_BETWEEN_CALLS,
        explanation=explanation, explanation_format=explanation_format,
        narrative=narrative,
    )
    completeness = int(comp.assessment)

    fl = llamar_con_pausa(
        _fluency_predict, grader_lm, GRADER_SECONDS_BETWEEN_CALLS,
        exemplars=exemplars, narrative=narrative,
    )
    fluency = int(fl.assessment)

    conciseness = conciseness_score(narrative, num_features=num_features, l_max=CONCISENESS_L_MAX)

    gd = llamar_con_pausa(
        _gdpr_predict, grader_lm, GRADER_SECONDS_BETWEEN_CALLS,
        ground_truth=ground_truth, narrative=narrative,
    )
    gdpr = gdpr_final_score(completeness, int(gd.assessment))

    return {
        "accuracy": accuracy,
        "completeness": completeness,
        "fluency": fluency,
        "conciseness": conciseness,
        "gdpr": gdpr,
        "justificaciones": {
            "accuracy": acc.justificacion,
            "completeness": comp.justificacion,
            "fluency": fl.justificacion,
            "gdpr": gd.justificacion,
        },
    }
