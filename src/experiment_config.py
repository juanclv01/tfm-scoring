"""
Configuracion central de la experimentacion Explingo (H x B).

Centraliza todo lo que debe permanecer IDENTICO entre ejecuciones separadas
en distintos dias (por el limite de free tier): modelos LLM, rutas de
checkpoint, la rejilla de configuraciones H/B de la tabla objetivo, y el
hiperparametro de conciseness ya calibrado.
"""
import os

# --------------------------------------------------------------------------
# Modelos LLM
# --------------------------------------------------------------------------
# NARRATOR: Nemotron 3 Ultra via NVIDIA Build directo (modelo
# "nvidia/nemotron-3-ultra-550b-a55b", confirmado en build.nvidia.com).
# Se mantiene pese a ser mas lento que Groq en el free tier compartido --
# decision de Juan de conservar Nemotron/Gemma en vez de cambiar a
# Llama/otros modelos.
NARRATOR_MODEL = os.environ.get("NARRATOR_MODEL", "openai/nvidia/nemotron-3-ultra-550b-a55b")
NARRATOR_API_KEY = os.environ.get("NVIDIA_API_KEY")
NARRATOR_API_BASE = "https://integrate.api.nvidia.com/v1"

# GRADER: Gemma 4 31B via NVIDIA Build directo -- sin cambios, ya validado
# (5/5 candidatos bootstrapped aceptados sin errores ni truncamiento).
GRADER_MODEL = os.environ.get("GRADER_MODEL", "openai/google/gemma-4-31b-it")
GRADER_API_KEY = os.environ.get("NVIDIA_API_KEY")
GRADER_API_BASE = "https://integrate.api.nvidia.com/v1"

# NARRATOR (Nemotron): temperatura moderada-alta -- da margen a fluidez y
# naturalidad del texto explicativo (dimension fluency de Explingo se
# resentiria con una narrativa demasiado mecanica/repetitiva a temp. 0).
# GRADER (Gemma): temperatura 0 -- el juicio de evaluacion debe ser
# determinista y consistente entre las 20 instancias y entre reintentos;
# variabilidad aqui introduciria ruido no deseado en las medias/desviaciones
# de la tabla comparativa.
NARRATOR_TEMPERATURE = 0.7
GRADER_TEMPERATURE = 0.0

# --------------------------------------------------------------------------
# Limites de tasa (free tier) -- pausas conservadoras entre llamadas.
# Ajustar tras confirmar los limites exactos en la documentacion de cada
# proveedor el dia de la ejecucion (ver nota en NARRATOR_MODEL arriba).
# --------------------------------------------------------------------------
# NVIDIA Build: limite de 40 req/min POR CUENTA, compartido entre NARRATOR
# y GRADER (mismo NVIDIA_API_KEY para ambos). 2s de pausa en cada uno deja
# margen holgado incluso si las llamadas se solapan en el tiempo (maximo
# teorico combinado ~30/min con esta pausa, por debajo del limite de 40).
NARRATOR_SECONDS_BETWEEN_CALLS = 2.0
GRADER_SECONDS_BETWEEN_CALLS = 2.0
LLM_REQUEST_TIMEOUT_SECONDS = 90     # timeout por peticion -- el default de litellm puede ser
                                       # demasiado corto para backends con arranque en frio

# --------------------------------------------------------------------------
# Seleccion de instancias del test set. Explingo (2024) usa 30 instancias
# diversas; se reduce aqui a 20 para bajar el volumen total de llamadas
# LLM del GRADER (20 x 4 dimensiones x 11 configs = 880, en vez de 1.320)
# y encajar en el plazo de una semana dado el limite de cuota gratuita
# disponible. Deben ser siempre las mismas 20 durante toda la experimentacion.
# --------------------------------------------------------------------------
N_TEST_INSTANCES = 20
TEST_INSTANCES_SEED = 42
TEST_INSTANCES_FILE = "test_instances_20.json"  # nombre distinto del fichero
                                                   # de 30 anterior -- evita
                                                   # mezclar selecciones

# --------------------------------------------------------------------------
# Rejilla de configuraciones H x B (orden = orden de la tabla adjunta por
# Juan). H y B se toman de forma ANIDADA: la configuracion H=3 usa las 3
# primeras narrativas hand-written de la lista fija (no 3 distintas cada
# vez), y B=3 usa los 3 primeros ejemplares bootstrapped generados para
# cada feature dominante. Esto es necesario para que el efecto de anadir
# MAS ejemplares sea aislable (RQ4 del TFM) -- si cada configuracion usara
# un subconjunto distinto, la comparacion entre filas de la tabla mezclaria
# dos variables (cuantos ejemplares + cuales ejemplares).
CONFIGURACIONES_HB = [
    (0, 0), (1, 0), (3, 0), (5, 0),
    (1, 1), (1, 3), (3, 1), (3, 3), (5, 1), (5, 3), (5, 5),
]

MAX_H = 5
MAX_B = 5

# --------------------------------------------------------------------------
# Conciseness (determinista, ya calibrado por Juan -- ver conciseness.py)
# --------------------------------------------------------------------------
CONCISENESS_L_MAX = 30

# --------------------------------------------------------------------------
# Umbral de aceptacion para el bootstrapping (Seccion 11 del contexto del
# nodo 2: un candidato bootstrapped solo se acepta como ejemplar si el
# propio GRADER lo valida en accuracy y completeness).
# --------------------------------------------------------------------------
BOOTSTRAP_ACCURACY_MIN = 1     # accuracy debe ser 1 (sin errores)
BOOTSTRAP_COMPLETENESS_MIN = 1  # completeness >= 1 (todos los features mencionados)
# NOTA: deliberadamente SIN umbral de GDPR aqui -- ver docstring de
# bootstrap_generator.py. GDPR es la variable que la rejilla H x B debe
# medir libremente (RQ4), no algo que el propio pool de ejemplares B deba
# garantizar de antemano.
BOOTSTRAP_CANDIDATE_POOL_FILE = "bootstrap_candidate_pool.json"
BOOTSTRAP_EXAMPLES_FILE = "bootstrapped_examples.json"

# --------------------------------------------------------------------------
# Checkpoint de resultados -- permite parar y reanudar la experimentacion
# entre sesiones sin recomputar nada ya hecho.
# --------------------------------------------------------------------------
RESULTS_CHECKPOINT_FILE = "resultados_experimento.json"
RESULTS_TABLE_FILE = "tabla_resultados.md"

# --------------------------------------------------------------------------
# Ponderacion de la "Puntuacion total" (columna final de la tabla).
# G = alpha_a*A + alpha_f*F + alpha_c*C + alpha_s*S + alpha_g*GDPR, con cada
# dimension ya normalizada a [0,1] (ver ESCALAS_MAX_DIMENSIONES). Decision
# de Juan: se prioriza accuracy y GDPR (los dos requisitos no negociables
# de fidelidad/cumplimiento normativo) por encima de fluency/conciseness
# (preferencias de estilo/UX).
# --------------------------------------------------------------------------
PESOS_DIMENSIONES = {
    "accuracy": 0.3,
    "completeness": 0.2,
    "fluency": 0.1,
    "conciseness": 0.1,
    "gdpr": 0.3,
}
ESCALAS_MAX_DIMENSIONES = {
    "accuracy": 1,
    "completeness": 2,
    "fluency": 4,
    "conciseness": 4,
    "gdpr": 3,
}


def config_key(h: int, b: int) -> str:
    """Clave textual estable para indexar resultados por configuracion."""
    return f"H{h}_B{b}"
