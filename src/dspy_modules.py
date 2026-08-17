"""
Modulos DSPy: un Signature por prompt ya cerrado (prompt_narr.txt,
prompt_grader_acc.txt, prompt_grader_comp.txt, prompt_grader_fl.txt,
prompt_grader_gdpr.txt), mas la configuracion de los dos LM (NARRATOR y
GRADER, ambos via NVIDIA Build directo -- ver experiment_config.py).

DECISION DE DISENO: no se usa dspy.teleprompt.BootstrapFewShot para el
few-shot. En su lugar, los demos (H y B) se asignan manualmente a
predict.demos (ver narrativas_hand_written.obtener_demos_narrator y
bootstrap_generator.py). Motivo: BootstrapFewShot no da control fino sobre
CUALES ejemplos concretos entran en cada configuracion ni sobre el
checkpointing entre llamadas -- ambas cosas son requisitos explicitos de
esta experimentacion (misma seleccion siempre; poder pausar/reanudar sin
volver a gastar cuota). El uso de DSPy sigue siendo real: Signature,
Predict, Example y LM son todos de la libreria.
"""
import re
import time

import dspy

from experiment_config import (
    NARRATOR_MODEL, GRADER_MODEL, NARRATOR_API_KEY, GRADER_API_KEY,
    NARRATOR_API_BASE, GRADER_API_BASE,
    NARRATOR_TEMPERATURE, GRADER_TEMPERATURE,
    NARRATOR_SECONDS_BETWEEN_CALLS, GRADER_SECONDS_BETWEEN_CALLS,
    LLM_REQUEST_TIMEOUT_SECONDS,
)


# ---------------------------------------------------------------------
# Saneado de la narrativa generada -- DSPy usa marcadores de protocolo
# internos (p.ej. "[[ ## completed ## ]]") para saber donde termina la
# respuesta estructurada del LM; normalmente el adapter los recorta antes
# de asignar el valor al campo de salida, pero se ha observado que algunos
# LM (confirmado con Nemotron) a veces los pegan sin el separador que el
# adapter espera, y el marcador se cuela como texto literal dentro de
# 'narrative'. Ninguna dimension del GRADER esta disenada para detectar
# artefactos de formato ajenos al contenido (accuracy/completeness solo
# verifican datos; fluency/conciseness no miran marcadores de protocolo),
# asi que si no se sanea aqui, pasa desapercibido -- y si esa narrativa se
# usa como ejemplar B, el NARRATOR podria aprender a imitar el artefacto.
# TODOS los puntos donde se captura narrator_out.narrative deben pasar por
# esta funcion antes de usar el texto para nada mas (grader, checkpoint,
# pool de bootstrapping).
# ---------------------------------------------------------------------
_PATRON_MARCADOR_DSPY = re.compile(r"\[\[\s*##.*?##\s*\]\]", re.IGNORECASE | re.DOTALL)

# CUIDADO: en espanol el punto TAMBIEN se usa como separador de MILES
# (p.ej. "1.542 DM" = mil quinientos cuarenta y dos, no un decimal). El
# patron exige EXACTAMENTE un digito tras el punto y ningun digito mas
# a continuacion -- coincide con "79.8" (formato real de shap_value,
# siempre con 1 decimal) pero nunca con "1.542" (3 digitos tras el punto).
_PATRON_DECIMAL_PUNTO = re.compile(r"(?<!\d)(\d+)\.(\d)(?!\d)")


def normalizar_decimales(texto: str) -> str:
    """Convierte notacion decimal de punto a coma (79.8 -> 79,8), sin
    tocar separadores de miles (1.542 se deja intacto). Red de seguridad
    determinista: el NARRATOR recibe instruccion de usar coma decimal en
    el prompt, pero eso no esta garantizado por codigo -- ninguna
    dimension del GRADER lo verifica, asi que si el LM la ignora, esta
    funcion lo corrige igualmente."""
    return _PATRON_DECIMAL_PUNTO.sub(r"\1,\2", texto)


def limpiar_narrativa(texto: str) -> str:
    """Elimina cualquier marcador de protocolo interno de DSPy (del tipo
    '[[ ## completed ## ]]' o similar) que se haya filtrado en el texto
    generado, normaliza la notacion decimal a coma, y recorta espacios/
    saltos de linea sobrantes resultantes."""
    limpio = _PATRON_MARCADOR_DSPY.sub("", texto)
    limpio = normalizar_decimales(limpio)
    return limpio.strip()
# ---------------------------------------------------------------------
# LM: dos instancias independientes. Comparten proveedor de HOSTING
# (NVIDIA Build) pero son modelos de LABORATORIOS distintos (NVIDIA
# Nemotron vs Google Gemma) -- se preserva el requisito de diseno de
# "proveedores distintos, sin puntos ciegos compartidos".
# ---------------------------------------------------------------------
def construir_lm_narrator() -> dspy.LM:
    kwargs = dict(
        api_key=NARRATOR_API_KEY,
        temperature=NARRATOR_TEMPERATURE,
        max_tokens=1500,
        num_retries=5,
        timeout=LLM_REQUEST_TIMEOUT_SECONDS,
        # extra_body/chat_template_kwargs -- confirmado por el propio
        # ejemplo de codigo de NVIDIA en build.nvidia.com para este modelo:
        # Nemotron 3 Ultra activa razonamiento por defecto
        # (enable_thinking=True en su sample). Se desactiva explicitamente
        # para evitar el truncamiento por tokens de pensamiento ya visto
        # con Qwen 3.6. max_tokens=1500 (mayor que el resto de LM) da
        # margen extra porque, incluso con enable_thinking=False, Nemotron
        # al ser un modelo mucho mayor tiende a generar respuestas algo
        # mas largas que modelos pequenos como llama-3.1-8b-instant.
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    if NARRATOR_API_BASE:
        kwargs["api_base"] = NARRATOR_API_BASE
    return dspy.LM(NARRATOR_MODEL, **kwargs)


def construir_lm_grader() -> dspy.LM:
    return dspy.LM(
        GRADER_MODEL,
        api_key=GRADER_API_KEY,
        api_base=GRADER_API_BASE,
        temperature=GRADER_TEMPERATURE,
        max_tokens=1200,  # subido de 800: cada dimension ahora devuelve
                          # tambien una justificacion textual (1-2 frases)
                          # ademas del digito, antes solo era el digito.
        num_retries=5,
        timeout=LLM_REQUEST_TIMEOUT_SECONDS,
        # Gemma 4 31B en NVIDIA Build lista "Reasoning: Supported" como
        # capacidad -- no confirmado si viene activado por defecto como en
        # Nemotron. Se pasa el mismo extra_body de forma preventiva: si el
        # backend de Gemma no reconoce chat_template_kwargs, lo ignora sin
        # error (va en el cuerpo de la peticion, no como parametro
        # validado por litellm, a diferencia de reasoning_effort que si
        # fallo con Groq).
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )


# ---------------------------------------------------------------------
# NARRATOR -- prompt_narr.txt
# ---------------------------------------------------------------------
class NarratorSignature(dspy.Signature):
    """You are helping users understand a Machine Learning model's
    prediction. Given an explanation and information about the model,
    convert the explanation into a human-readable narrative.

    The Decision, Risk level and Score are already final -- do not infer,
    recompute, or contradict them from the Explanation; they may not follow
    obviously from summing the signs of the factors listed, since the
    Explanation lists only the top 5 most influential factors out of many
    more that also contributed to the decision. MANDATORY: the narrative
    must explicitly state the exact Score number -- it is a fundamental
    part of grounding the explanation for the client, not optional detail.
    It must match the given value exactly; never state an approximate or
    invented number.

    If a factor's direction seems counterintuitive from a conventional
    financial standpoint (e.g. an established credit history lowering the
    score, or the absence of a savings account raising it), report it
    factually and neutrally exactly as given, without inventing a causal
    justification for why it occurs. Do not mention proba_default or any
    decision threshold; those are internal and not part of the Explanation.

    MANDATORY: do not use Markdown formatting of any kind -- no **bold**,
    no *italics*, no bullet or numbered lists, no headers. Write plain
    prose paragraphs only, exactly as natural written Spanish business
    correspondence would look.

    MANDATORY: use Spanish-locale decimal notation (comma as decimal
    separator, e.g. "79,8 puntos") for every point value you state from
    the Explanation. The Explanation itself uses a period as decimal
    separator (standard programmatic number formatting, e.g. "+79.8 pts")
    -- that is only the raw data format, never copy it verbatim; always
    convert it to the Spanish comma notation in the Narrative.

    Write the Narrative in Spanish."""

    context: str = dspy.InputField()
    decision: str = dspy.InputField(desc="APROBADA or RECHAZADA")
    risk_level: str = dspy.InputField(desc="bajo, moderado, or alto")
    score: int = dspy.InputField(desc="the client's credit score, 0-1000 scale, already computed -- MUST be stated explicitly in the narrative")
    explanation: str = dspy.InputField()
    explanation_format: str = dspy.InputField()
    narrative: str = dspy.OutputField(desc="human-readable narrative, in Spanish, plain prose, no Markdown, decimal comma notation")


# ---------------------------------------------------------------------
# GRADER: accuracy -- prompt_grader_acc.txt
# ---------------------------------------------------------------------
class AccuracySignature(dspy.Signature):
    """Assess a narrative's accuracy against a rubric.

    Check three things: (1) the narrative's stated decision and risk level
    must match Ground truth exactly; (1b) the narrative MUST explicitly
    state the exact Score number from Ground truth -- omitting it entirely
    is an error, and so is stating a different, approximate, or rounded
    number; (2) for every factor mentioned, its value and contribution
    direction must match the Explanation (a positive shap_value RAISES the
    score, a negative shap_value LOWERS it).

    Do NOT penalize a factor merely because its direction seems
    counterintuitive from a conventional standpoint (e.g. an established
    credit history lowering the score); that is a real, documented effect
    of this dataset, not an error. Only mark it as an error if the
    narrative's stated direction contradicts the Explanation itself, or if
    the narrative invents an unsupported causal reason for why the effect
    occurs.

    Do NOT treat decimal notation differences as an error: the Explanation
    uses a period as decimal separator (e.g. "79.8") while the Narrative is
    expected to use Spanish comma notation (e.g. "79,8") -- these represent
    the SAME number and must be judged as equivalent, never as a mismatch.

    feature_raw/codigo_ohe in the Explanation are only for cross-checking
    that feature_value truthfully reflects the client's data -- never judge
    the narrative for not mentioning them (it never should).

    Rubric: 0 = contains one or more errors in value, contribution
    direction, in the stated decision/risk level, or in the score number
    (including omitting it entirely). 1 = contains no errors and states
    the correct score, but may be missing other (non-mandatory)
    information.

    Respond with your reasoning in `justificacion` first, then the digit in
    `assessment`."""

    ground_truth: str = dspy.InputField()
    explanation: str = dspy.InputField()
    explanation_format: str = dspy.InputField()
    narrative: str = dspy.InputField()
    justificacion: str = dspy.OutputField(
        desc="1-2 sentences in Spanish explaining exactly why this score was given "
             "(which specific error was found, or confirmation that everything matched). "
             "FOR THE RESEARCHER ONLY -- never shown to the client."
    )
    assessment: int = dspy.OutputField(desc="0 or 1, single digit only")


# ---------------------------------------------------------------------
# GRADER: completeness -- prompt_grader_comp.txt
# ---------------------------------------------------------------------
class CompletenessSignature(dspy.Signature):
    """Assess how completely a narrative describes an explanation.

    Rubric: 0 = one or more feature names from the explanation are not
    mentioned at all in the narrative. 1 = all features are mentioned, but
    not all feature values and/or contribution directions. 2 = all features
    are mentioned, and for each feature, includes at least an approximation
    of the feature's value and contribution direction (direction need not
    be verified for correctness here -- that is assessed separately by
    accuracy).

    Respond with your reasoning in `justificacion` first, then the digit in
    `assessment`."""

    explanation: str = dspy.InputField()
    explanation_format: str = dspy.InputField()
    narrative: str = dspy.InputField()
    justificacion: str = dspy.OutputField(
        desc="1-2 sentences in Spanish explaining exactly why this score was given "
             "(which feature(s) are missing or incomplete, or confirmation that all "
             "are present). FOR THE RESEARCHER ONLY -- never shown to the client."
    )
    assessment: int = dspy.OutputField(desc="0, 1 or 2, single digit only")


# ---------------------------------------------------------------------
# GRADER: fluency -- prompt_grader_fl.txt
# ---------------------------------------------------------------------
class FluencySignature(dspy.Signature):
    """Assess how well the style of a narrative matches the style of
    example narratives. Consider only the linguistic style (word choice,
    sentence structure, tone, register), not the topic or the specific
    features/values discussed. The narrative being evaluated is written in
    Spanish; judge it against the exemplars on its own terms as
    Spanish-language text, not by comparison to English style norms.

    Rubric: 0 = very dissimilar. 1 = dissimilar. 2 = neutral. 3 = similar.
    4 = very similar.

    Respond with your reasoning in `justificacion` first, then the digit in
    `assessment`. If Exemplars is empty, judge the narrative's style on its
    own general fluency/naturalness in Spanish instead."""

    exemplars: str = dspy.InputField(desc="may be empty if H=0")
    narrative: str = dspy.InputField()
    justificacion: str = dspy.OutputField(
        desc="1-2 sentences in Spanish explaining exactly why this score was given "
             "(what specifically matches or differs in style/register from the "
             "exemplars, or general fluency observations if Exemplars is empty). "
             "FOR THE RESEARCHER ONLY -- never shown to the client."
    )
    assessment: int = dspy.OutputField(desc="0-4, single digit only")


# ---------------------------------------------------------------------
# GRADER: GDPR (a)+(b) -- prompt_grader_gdpr.txt
# ---------------------------------------------------------------------
class GdprSignature(dspy.Signature):
    """Assess whether a narrative satisfies two GDPR Article 22
    requirements for automated-decision explanations. (A third requirement
    -- that the main determining factors are stated -- is assessed
    separately, deterministically, and is NOT part of this evaluation.)

    (a) Sense of the decision: does the narrative clearly and correctly
    convey what the decision is (approved/rejected) and, in general terms,
    the direction in which the listed factors moved the outcome, in a way
    an ordinary client could understand without technical knowledge of SHAP
    or machine learning? Check this against Ground truth: a narrative that
    states or implies a decision inconsistent with Ground truth fails this
    requirement, even if the rest of the text is fluent and well-organized.

    (b) Right to contest: does the narrative explicitly inform the client
    that they have the right to express their point of view and to request
    human review of / contest the automated decision?

    Rubric: 0 = neither (a) nor (b) is satisfied. 1 = exactly one of (a) or
    (b) is satisfied. 2 = both (a) and (b) are satisfied.

    Respond with your reasoning in `justificacion` first, then the digit in
    `assessment`."""

    ground_truth: str = dspy.InputField()
    narrative: str = dspy.InputField()
    justificacion: str = dspy.OutputField(
        desc="1-2 sentences in Spanish explaining exactly why this score was given "
             "(whether (a) sense of decision and (b) right to contest were each "
             "satisfied, and why). FOR THE RESEARCHER ONLY -- never shown to the client."
    )
    assessment: int = dspy.OutputField(desc="0, 1 or 2, single digit only")


def llamar_con_pausa(predict_module: dspy.Predict, lm: dspy.LM, seconds_between_calls: float, **kwargs):
    """Ejecuta un modulo dspy.Predict bajo un LM concreto, respetando una
    pausa fija tras la llamada para no exceder el limite de RPM del free
    tier. dspy.LM ya reintenta con backoff en fallos transitorios (ver
    num_retries en construir_lm_*); ademas, ante un 429 de pool compartido
    de OpenRouter (RateLimitError con mensaje "temporarily rate-limited
    upstream"), se aplica aqui un backoff mas largo y deliberado antes de
    propagar el error -- ese tipo de saturacion suele resolverse en
    segundos/minutos, no requiere abandonar el candidato/instancia."""
    esperas_pool_compartido = [10, 30, 60]  # segundos, backoff creciente
    ultimo_error = None
    for espera in [0] + esperas_pool_compartido:
        if espera:
            print(f"    (pool compartido saturado, esperando {espera}s antes de reintentar...)")
            time.sleep(espera)
        try:
            with dspy.context(lm=lm):
                resultado = predict_module(**kwargs)
            time.sleep(seconds_between_calls)
            return resultado
        except Exception as exc:
            ultimo_error = exc
            nombre_tipo = type(exc).__name__
            es_transitorio = (
                "RateLimitError" in nombre_tipo or "429" in str(exc)
                or "Timeout" in nombre_tipo or "aborted" in str(exc).lower()
            )
            if not es_transitorio:
                raise  # error distinto a saturacion/timeout -- no reintentar aqui, dejar que suba
    raise ultimo_error
