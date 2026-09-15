"""
Nodo 3 (Narrador), Nodo 4 (Evaluador) y el nodo de revision humana, mas
la funcion de enrutamiento que implementa el guardarraiz de reintento
como arista condicional de LangGraph.

# CORRECTED: el guardarraiz de reintento se implemento inicialmente en
# app_backend.py como un bucle 'for' imperativo que llamaba directamente
# a los componentes DSPy, sin pasar por el grafo compilado -- una
# desviacion del diseno original (ver justificacion de LangGraph por su
# capacidad de expresar aristas condicionales, [REF]) documentada en su
# momento como tal. Se corrige aqui: node_narrative() y node_grader()
# son nodos del mismo StateGraph que ya orquesta los Nodos 1-2, y
# decide_after_grader() es la funcion de enrutamiento que
# graph.add_conditional_edges() usa para decidir, tras cada evaluacion,
# si se reintenta, se aprueba o se escala a revision humana -- sin
# ningun bucle Python explicito: es el propio grafo el que vuelve a
# ejecutar el Nodo 3 cuantas veces haga falta, hasta MAX_INTENTOS.
#
# No se reimplementa aqui ninguna logica ya validada en la
# experimentacion offline: NarratorSignature, construir_lm_narrator(),
# construir_lm_grader(), evaluar_narrativa() y puntuacion_total() son
# exactamente los mismos componentes que usan run_experiment.py y
# bootstrap_generator.py.
"""
import dspy

from graph_state import PipelineState
from narrativas_hand_written import (
    CONTEXT, EXPLANATION_FORMAT, obtener_demos_narrator, NARRATIVAS_HAND_WRITTEN,
)
from dspy_modules import NarratorSignature, construir_lm_narrator, construir_lm_grader, limpiar_narrativa
from metrics import evaluar_narrativa
from build_results_table import puntuacion_total
from pipeline_helpers import formatear_explanation, construir_ground_truth


# ---------------------------------------------------------------------
# Parametros del guardarraiz de reintento (Nodo 4)
# ---------------------------------------------------------------------
MAX_INTENTOS = 5  # tal como se especifico originalmente: hasta 5 intentos antes de escalar
VALIDACION_ACCURACY_REQUERIDA = 1  # ningun error factual tolerado, sin excepcion
VALIDACION_PUNTUACION_MINIMA = 0.85  # ver aviso de calibracion mas abajo

# AVISO DE CALIBRACION: 0.85 es un valor de partida razonable, no un
# umbral ya validado empiricamente contra los resultados de la
# experimentacion H x B -- ajustar una vez se disponga de la tabla de
# resultados completa y se observe que puntuacion total obtienen en la
# practica las narrativas consideradas "buenas".


# ---------------------------------------------------------------------
# Signature del NARRADOR extendida con el motivo del reintento.
# EXTIENDE NarratorSignature via .append() -- no la modifica, no
# duplica su docstring/instrucciones (se heredan automaticamente), y no
# afecta a los scripts de experimentacion (run_experiment.py,
# bootstrap_generator.py siguen usando NarratorSignature tal cual, sus
# checkpoints no se ven afectados).
# ---------------------------------------------------------------------
NarratorConReintentoSignature = NarratorSignature.append(
    "motivo_reintento",
    dspy.InputField(
        desc="If this is a retry after a previous rejection by the quality "
             "reviewer, this briefly explains in Spanish why it was rejected "
             "so the new narrative can correct it. Empty string on the first "
             "attempt."
    ),
    type_=str,
)


# ---------------------------------------------------------------------
# Configuracion FIJA del NARRADOR para la app: H=1, B=0.
# Inicializacion perezosa a nivel de modulo, mismo patron que _model/
# _explainer en nodes.py para los Nodos 1-2.
# ---------------------------------------------------------------------
_narrator_predict = dspy.Predict(NarratorConReintentoSignature)
_narrator_predict.demos = obtener_demos_narrator(1)  # H=1, B=0 (fijo, sin exemplares bootstrapped)
_lm_narrator = construir_lm_narrator()
_lm_grader = construir_lm_grader()

_EXEMPLAR_FLUENCY = NARRATIVAS_HAND_WRITTEN[0]["narrative"]  # mismo H=1 usado como demo del NARRATOR


def _construir_motivo_reintento(resultado_grader: dict) -> str:
    """Resume, en espanol, por que se rechazo la narrativa anterior --
    se le pasa al Nodo 3 en el siguiente intento (reintento informado,
    no repetir el mismo prompt a ciegas)."""
    justif = resultado_grader["justificaciones"]
    razones = []
    if resultado_grader["accuracy"] < VALIDACION_ACCURACY_REQUERIDA:
        razones.append(f"Exactitud insuficiente: {justif['accuracy']}")
    if resultado_grader["completeness"] < 2:
        razones.append(f"Completitud mejorable: {justif['completeness']}")
    if resultado_grader["gdpr"] < 3:
        razones.append(f"Cumplimiento GDPR mejorable: {justif['gdpr']}")
    if not razones:
        razones.append(f"Puntuación total por debajo del umbral requerido "
                        f"({VALIDACION_PUNTUACION_MINIMA}).")
    return " ".join(razones)


def node_narrative(state: PipelineState) -> dict:
    """
    NODO 3 -- Narrador. Genera (o regenera, si motivo_reintento no esta
    vacio) la narrativa a partir de las 5 tuplas SHAP ya decodificadas
    por el Nodo 2.

    LangGraph vuelve a ejecutar este nodo completo cada vez que la
    arista condicional del Nodo 4 (decide_after_grader) decide "retry"
    -- no hay ningun bucle explicito aqui, el propio grafo es el bucle.
    Por eso el contador de intentos se incrementa dentro del propio
    nodo y se persiste en el estado compartido, no en una variable local
    de Python que se perderia entre invocaciones del nodo.
    """
    explanation = formatear_explanation(state["top_features"])
    motivo_reintento = state.get("motivo_reintento", "")

    with dspy.context(lm=_lm_narrator):
        narrator_out = _narrator_predict(
            context=CONTEXT,
            decision="APROBADA" if state["aprobado"] else "RECHAZADA",
            risk_level=state["nivel_riesgo"],
            score=state["score"],
            explanation=explanation,
            explanation_format=EXPLANATION_FORMAT,
            motivo_reintento=motivo_reintento,
        )
    narrativa = limpiar_narrativa(narrator_out.narrative)

    return {
        "narrative": narrativa,
        "intentos": state.get("intentos", 0) + 1,
    }


def node_grader(state: PipelineState) -> dict:
    """
    NODO 4 -- Evaluador. Puntua la narrativa del Nodo 3 en las 5
    dimensiones y determina si supera el umbral de aprobacion.

    Este nodo NO decide por si mismo si se reintenta, se aprueba o se
    escala -- esa decision es responsabilidad exclusiva de la arista
    condicional decide_after_grader(), que LangGraph invoca justo
    despues de este nodo. node_grader() se limita a calcular y devolver
    validation_result; separar "evaluar" de "decidir el siguiente paso"
    es lo que permite expresar el guardarraiz como arista condicional
    en lugar de como logica interna de un unico nodo monolitico.
    """
    ground_truth = construir_ground_truth(state)
    explanation = formatear_explanation(state["top_features"])

    resultado_grader = evaluar_narrativa(
        narrative=state["narrative"],
        ground_truth=ground_truth,
        explanation=explanation,
        explanation_format=EXPLANATION_FORMAT,
        exemplars=_EXEMPLAR_FLUENCY,
        num_features=len(state["top_features"]),
        grader_lm=_lm_grader,
    )

    accuracy_ok = resultado_grader["accuracy"] >= VALIDACION_ACCURACY_REQUERIDA
    puntuacion = puntuacion_total(resultado_grader)
    aprobada = accuracy_ok and puntuacion >= VALIDACION_PUNTUACION_MINIMA

    validation_result = {
        "aprobado": aprobada,
        "puntuacion_total": puntuacion,
        "detalle": resultado_grader,
        "motivo": None if aprobada else _construir_motivo_reintento(resultado_grader),
    }

    salida = {"validation_result": validation_result}
    if not aprobada:
        # Se persiste ya aqui el motivo_reintento (en vez de calcularlo
        # de nuevo en node_narrative) para que decide_after_grader() y
        # el propio Nodo 3, en su siguiente ejecucion, lean el mismo
        # valor sin duplicar la logica de construccion del mensaje.
        salida["motivo_reintento"] = validation_result["motivo"]
    return salida


def decide_after_grader(state: PipelineState) -> str:
    """
    Arista condicional del Nodo 4 -- el unico punto de bifurcacion
    condicional de todo el grafo ([REF], justificacion de por que un
    pipeline secuencial no basta y se eligio LangGraph). Tres
    desenlaces posibles:

      "approved": la narrativa supera el umbral -> fin del grafo.
      "retry":    no lo supera y quedan intentos -> vuelve al Nodo 3.
      "escalate": no lo supera y se agotaron los MAX_INTENTOS ->
                  nodo de revision humana.
    """
    if state["validation_result"]["aprobado"]:
        return "approved"
    if state["intentos"] >= MAX_INTENTOS:
        return "escalate"
    return "retry"


def node_human_review(state: PipelineState) -> dict:
    """
    Nodo de escalado -- plantilla FIJA, SIN LLM. Si el sistema
    generativo ya ha fallado MAX_INTENTOS veces en superar el umbral de
    calidad, no tiene sentido confiar en el mismo sistema para redactar
    el propio aviso de fallo.
    """
    mensaje = (
        f"Su solicitud ha sido evaluada (puntuación: {state['score']}/1000). "
        "No ha sido posible generar automáticamente una explicación que "
        "cumpla nuestros estándares de calidad. En cumplimiento del "
        "artículo 22 del RGPD, un responsable humano revisará su caso. "
        "Tiene derecho a expresar su punto de vista y a impugnar esta decisión."
    )
    return {"narrative": mensaje}
