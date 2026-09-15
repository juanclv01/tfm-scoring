"""
Backend FastAPI del pipeline de scoring crediticio explicable.

# CORRECTED: la version anterior de este modulo invocaba directamente
# los componentes DSPy del NARRADOR y del EVALUADOR desde un bucle
# Python (for intento in range(1, MAX_INTENTOS + 1): ...), sin pasar en
# ningun momento por el grafo compilado de graph.py -- una desviacion
# del diseno original que este TFM justifica explicitamente por la
# necesidad de una arista condicional ([REF]). Se corrige aqui: el
# guardarraiz de reintento del Nodo 4 vive ahora dentro del propio
# StateGraph (ver graph.py y nodes_narrator_grader.py), como arista
# condicional real. Este modulo se limita a:
#   1) construir el grafo compilado una unica vez, al importar el modulo;
#   2) construir el estado inicial a partir de los datos del cliente;
#   3) invocar el grafo (_grafo.invoke(...)) una vez por solicitud;
#   4) traducir el estado final del grafo a la respuesta de la API.
# No se reimplementa aqui ninguna logica de orquestacion, generacion ni
# evaluacion -- toda esa logica vive en los nodos del grafo.

Uso:
    uvicorn app_backend:app --reload --port 8000
"""
import os
import random
import time

import joblib  # numpy antes que dspy -- ver nota en run_experiment.py

import litellm
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from graph import build_graph
from select_test_instances import cargar_o_seleccionar_instancias
from pipeline_helpers import hash_client_data
from experiment_config import NARRATOR_MODEL, GRADER_MODEL
from nodes_narrator_grader import MAX_INTENTOS, VALIDACION_PUNTUACION_MINIMA

try:
    from langsmith import traceable
except ImportError:  # langsmith no instalado -- la app sigue funcionando sin trazas
    def traceable(*args, **kwargs):
        def decorador(fn):
            return fn
        return decorador


# ---------------------------------------------------------------------
# LangSmith: activar el callback nativo de litellm si hay credenciales.
# No se inventa ningun proyecto/clave -- si no esta configurado, se avisa
# por consola y la app sigue funcionando sin trazas remotas.
# ---------------------------------------------------------------------
_LANGSMITH_ACTIVO = bool(os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY"))
if _LANGSMITH_ACTIVO:
    litellm.success_callback = ["langsmith"]
    litellm.failure_callback = ["langsmith"]
    os.environ.setdefault("LANGCHAIN_PROJECT", os.environ.get("LANGSMITH_PROJECT", "tfm-scoring-explicable"))
    print("LangSmith activo -- las llamadas al NARRATOR y al GRADER se trazaran en el "
          f"proyecto '{os.environ.get('LANGCHAIN_PROJECT')}'.")
else:
    print("LANGSMITH_API_KEY no definida -- la app funciona igual, pero sin trazas "
          "remotas en LangSmith. Define LANGSMITH_API_KEY para activarlo.")


# ---------------------------------------------------------------------
# Grafo compilado (Nodos 1-4 + guardarraiz de reintento como arista
# condicional) e instancias de demo, ambos inicializados una unica vez.
# ---------------------------------------------------------------------
_grafo = build_graph()
_INSTANCIAS_TEST = None


def _instancias_test() -> list:
    global _INSTANCIAS_TEST
    if _INSTANCIAS_TEST is None:
        _INSTANCIAS_TEST = cargar_o_seleccionar_instancias()
    return _INSTANCIAS_TEST


# ---------------------------------------------------------------------
# Modelos de datos
# ---------------------------------------------------------------------
class ClientData(BaseModel):
    """Los 19 campos crudos de German Credit -- exactamente los mismos
    que consume node_scoring()/node_explainability(), sin ningun campo
    inventado (p.ej. no hay TIN, no existe en el pipeline)."""
    checking_status: str
    duration: int
    credit_history: str
    purpose: str
    credit_amount: int
    savings_status: str
    employment: str
    installment_rate: int
    personal_status: str
    other_parties: str
    residence_since: int
    property_magnitude: str
    age: int
    other_payment_plans: str
    housing: str
    existing_credits: int
    job: str
    num_dependents: int
    own_telephone: str
    foreign_worker: str


class FactorSHAP(BaseModel):
    feature_name: str
    feature_value: str
    shap_value: float


class RespuestaScoring(BaseModel):
    id_solicitante: str = Field(description="Hash no identificativo del cliente (nunca su nombre)")
    aprobado: bool
    nivel_riesgo: str
    score: int
    factores: list[FactorSHAP]
    narrativa: str
    modelo_narrator: str
    modelo_grader: str
    configuracion_narrator: str = "H=1, B=0"
    tiempo_generacion_seg: float
    # --- Nodo 4: guardarraiz de reintento (arista condicional del grafo) ---
    narrativa_aprobada_por_grader: bool = Field(
        description="False si se agotaron los MAX_INTENTOS y se activo el mensaje de revision humana"
    )
    intentos_narrator: int = Field(description="Cuantas veces se ejecuto el Nodo 3 (1 si se aprobo a la primera)")
    requiere_revision_humana: bool


# ---------------------------------------------------------------------
# Logica compartida: de client_data a respuesta completa
# ---------------------------------------------------------------------
@traceable(name="pipeline_scoring_explicable")
def _generar_explicacion(client_data: dict) -> RespuestaScoring:
    """
    Invoca el grafo compilado una unica vez. El guardarraiz de
    reintento del Nodo 4 -- incluidos los reintentos al Nodo 3 y el
    posible escalado a revision humana -- ocurre enteramente DENTRO de
    esta llamada a _grafo.invoke(): no hay ningun bucle en este modulo.
    """
    inicio = time.time()

    estado_final = _grafo.invoke({"client_data": client_data})

    return RespuestaScoring(
        id_solicitante=hash_client_data(client_data),
        aprobado=estado_final["aprobado"],
        nivel_riesgo=estado_final["nivel_riesgo"],
        score=estado_final["score"],
        factores=[
            FactorSHAP(
                feature_name=f["feature_name"],
                feature_value=f["feature_value"],
                shap_value=f["shap_value"],
            )
            for f in estado_final["top_features"]
        ],
        narrativa=estado_final["narrative"],
        modelo_narrator=NARRATOR_MODEL,
        modelo_grader=GRADER_MODEL,
        tiempo_generacion_seg=round(time.time() - inicio, 2),
        narrativa_aprobada_por_grader=estado_final["validation_result"]["aprobado"],
        intentos_narrator=estado_final["intentos"],
        requiere_revision_humana=not estado_final["validation_result"]["aprobado"],
    )


# ---------------------------------------------------------------------
# API
# ---------------------------------------------------------------------
app = FastAPI(
    title="API de scoring crediticio explicable",
    description="Grafo LangGraph de 4 nodos: Nodo 1 (XGBoost) + Nodo 2 (SHAP) + "
                 "Nodo 3 (NARRATOR, H=1/B=0) + Nodo 4 (GRADER + arista condicional "
                 f"de reintento, hasta {MAX_INTENTOS} intentos).",
    version="2.0.0",
)


@app.post("/score", response_model=RespuestaScoring)
def score_cliente(datos: ClientData):
    """Funcion principal: recibe los datos de un solicitante y devuelve
    score, decision, factores SHAP y narrativa explicativa."""
    try:
        return _generar_explicacion(datos.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/demo/instancia_aleatoria", response_model=RespuestaScoring)
def demo_instancia_aleatoria():
    """Funcionalidad de demo: toma una instancia ALEATORIA de las 20 del
    conjunto de test ya usado en la experimentacion (test_instances_20.json
    -- nunca datos inventados) y genera su explicacion completa."""
    instancias = _instancias_test()
    elegida = random.choice(instancias)
    try:
        return _generar_explicacion(elegida["client_data"])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health")
def health():
    return {
        "status": "ok",
        "langsmith_activo": _LANGSMITH_ACTIVO,
        "modelo_narrator": NARRATOR_MODEL,
        "modelo_grader": GRADER_MODEL,
        "max_intentos_guardarraiz": MAX_INTENTOS,
        "validacion_puntuacion_minima": VALIDACION_PUNTUACION_MINIMA,
        "guardarraiz_implementado_como": "arista condicional de LangGraph (graph.py)",
    }
