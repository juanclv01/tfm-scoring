# Aplicación de scoring crediticio explicable

Backend FastAPI + frontend Streamlit sobre el pipeline completo:
Nodo 1 (XGBoost) → Nodo 2 (SHAP) → Nodo 3 (NARRATOR, H=1/B=0) →
**Nodo 4 (GRADER + guardarraíl de reintento, hasta 5 intentos)**.

## Guardarraíl de reintento (Nodo 4)

- Tras generar la narrativa, el GRADER la evalúa en las 5 dimensiones.
- Se considera **APROBADA** si `accuracy=1` (ningún error factual tolerado)
  **y** la Puntuación total ponderada (misma fórmula y pesos que
  `build_results_table.puntuacion_total()`, ya usada en la tabla de
  experimentación) alcanza `VALIDACION_PUNTUACION_MINIMA` (0.85 por
  defecto, en `app_backend.py`).
- Si se rechaza, se reintenta pasando al NARRATOR el motivo concreto del
  rechazo (extraído de las justificaciones del GRADER) — reintento
  informado, no se repite el mismo prompt a ciegas.
- Tras 5 intentos sin aprobar, se activa un mensaje de revisión humana:
  **plantilla fija, sin LLM** (si el sistema generativo ya falló 5 veces,
  no tiene sentido confiar en él para el propio aviso).

**⚠️ Aviso de calibración**: `VALIDACION_PUNTUACION_MINIMA=0.85` es un
valor de partida razonable, no un umbral ya validado empíricamente contra
tus resultados de experimentación. Ajústalo una vez tengas la tabla H1_B0
completa y veas qué puntuación total obtienen en la práctica las
narrativas que consideras "buenas".

**Coste**: cada intento del guardarraíl cuesta 1 llamada NARRATOR + 4
llamadas GRADER = 5 llamadas. En el peor caso (5 intentos, revisión
humana) son 25 llamadas por una sola consulta — ten en cuenta la
lentitud ya conocida de Nemotron en el free tier de NVIDIA Build.

## Instalación

```powershell
pip install -r requirements_app.txt
```

(el resto de dependencias del pipeline — joblib, pandas, scikit-learn,
xgboost, shap, dspy — ya deberías tenerlas instaladas)

## Variables de entorno

Las mismas que ya usas para la experimentación (`NVIDIA_API_KEY` como
mínimo), más, opcionalmente, para activar LangSmith:

```powershell
$env:NVIDIA_API_KEY = "tu_clave"
$env:LANGSMITH_API_KEY = "tu_clave_de_langsmith"      # opcional
$env:LANGSMITH_PROJECT = "tfm-scoring-explicable"      # opcional, nombre del proyecto en LangSmith
```

Si no defines `LANGSMITH_API_KEY`, la app funciona igual, solo que sin
trazas remotas (se avisa por consola al arrancar). Con LangSmith activo,
se trazan tanto las llamadas al NARRATOR como al GRADER (todos los
intentos del guardarraíl, no solo el aprobado).

## Ejecución (dos terminales)

**Terminal 1 — backend:**
```powershell
cd D:\Juan\tfm-scoring\src
uvicorn app_backend:app --reload --port 8000
```

**Terminal 2 — frontend:**
```powershell
cd D:\Juan\tfm-scoring\src
streamlit run app_frontend.py
```

Se abrirá en el navegador (por defecto `http://localhost:8501`). El
frontend habla con el backend en `http://localhost:8000` — si cambias el
puerto del backend, ajusta la variable `API_URL` antes de lanzar Streamlit:

```powershell
$env:API_URL = "http://localhost:8000"
```

## Los dos modos de la app

1. **Consultar solicitante**: formulario con los 19 campos crudos de
   German Credit (códigos Statlog, p. ej. `A11`-`A14`) → `POST /score`.
2. **Demo: instancia aleatoria**: botón que pide una instancia al azar de
   las 20 del conjunto de test ya usado en la experimentación
   (`test_instances_20.json` — nunca datos inventados) → `GET /demo/instancia_aleatoria`.

Ambos modos pasan por el guardarraíl completo del Nodo 4.

## Endpoints de la API

- `POST /score` — cuerpo: los 19 campos de `ClientData`. Devuelve score,
  decisión, nivel de riesgo, factores SHAP, narrativa, y el resultado del
  guardarraíl (`narrativa_aprobada_por_grader`, `intentos_narrator`,
  `requiere_revision_humana`).
- `GET /demo/instancia_aleatoria` — sin cuerpo, misma respuesta.
- `GET /health` — estado de la API, si LangSmith está activo, y los
  parámetros del guardarraíl (`max_intentos_guardarraiz`,
  `validacion_puntuacion_minima`).

Documentación interactiva automática en `http://localhost:8000/docs`
(Swagger UI de FastAPI).

## Notas de diseño

- **Sin nombre de cliente**: se muestra un hash de 12 caracteres
  (`pipeline_helpers.hash_client_data`) en vez del nombre — mismo mecanismo
  ya usado para verificar integridad del checkpoint en la experimentación.
- **Sin TIN**: no aparece porque no se calcula en ningún nodo del pipeline
  real — no se ha inventado.
- **Configuración NARRATOR fija en H=1, B=0**: se fija explícitamente al
  arrancar `app_backend.py` (`_narrator_predict.demos = obtener_demos_narrator(1)`),
  reutilizando el mismo ejemplar hand-written que usa la experimentación,
  no una copia nueva.
- **El campo `motivo_reintento` NO toca `NarratorSignature`**: se extiende
  vía `NarratorSignature.append(...)`, así que `run_experiment.py` y
  `bootstrap_generator.py` (y sus checkpoints ya generados) no se ven
  afectados por este cambio.

