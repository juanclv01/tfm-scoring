# TFM — Workflow Agéntico para Scoring Crediticio

Nodo 1 (XGBoost) + Nodo 2 (SHAP) orquestados con LangGraph.
Línea T-002 — IA y Agentes Inteligentes con LLMs.

## Estado del pipeline

```
[CSV cliente] -> Nodo 1 (XGBoost) -> Nodo 2 (SHAP) -> Nodo 3 (LLM) -> Nodo 4 (Validador GDPR)
                 [implementado]      [implementado]    [pendiente]     [pendiente]
```

## Setup

```bash
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell
# source .venv/bin/activate       # macOS / Linux

pip install --upgrade pip
pip install -r requirements.txt
```

## Datasets utilizados

| Dataset | Rol | Fuente | Ruta | Estado actual |
|---|---|---|---|---|
| German Credit | Prototipo inicial + modelo del pipeline en vivo | UCI | `data/german-credit-data/german.data` | **En uso** |
| Default Credit Card | Validación cruzada de métricas | UCI (.xls original) | `data/default-of-credit-card-clients/` | **En uso** |
| Home Credit Default | Validación final, prueba de escala | Kaggle | `data/home-credit-default/application_train.csv` | Diferido — código presente, no ejecutado todavía |

**Importante:** el grafo en vivo (`graph.py`, `nodes.py`) sigue alimentado únicamente
por German Credit. Los otros dos datasets son validaciones offline, no sustituyen
el modelo del pipeline de demo.

**Sobre Home Credit (diferido):** todo el código (`data_loader_home_credit.py`,
`train_model_home_credit.py`, `scripts/download_home_credit.py`) permanece en el
repositorio y es funcional, pero no forma parte del flujo de trabajo actual. El
pipeline **no depende** de que este dataset se haya entrenado: `evaluate_cross_dataset.py`
detecta si el `.joblib` de Home Credit no existe todavía y lo salta (imprime
"(modelo no entrenado aun)" en esa fila) sin fallar el resto de la ejecución.
Retómalo más adelante ejecutando únicamente los pasos de Home Credit de la
sección "Orden de ejecución" — no requiere ningún cambio de código.

**Descarga:**
- German Credit: ver `data/LEEME.txt`.
- Default Credit Card: descarga el `.xls` original de UCI y colócalo (el fichero,
  con cualquier nombre) dentro de la carpeta `data/default-of-credit-card-clients/`.
  `load_credit_card()` localiza automáticamente el `.xls`/`.xlsx` dentro de esa
  carpeta — no hace falta indicar el nombre exacto del fichero ni renombrarlo.
  Requiere `xlrd` (`.xls`) u `openpyxl` (`.xlsx`), ya incluidos en `requirements.txt`.
- Home Credit Default: competición de Kaggle, requiere **aceptar las reglas en
  el navegador primero** (https://www.kaggle.com/competitions/home-credit-default-risk/rules)
  — sin esto, cualquier descarga por API (kagglehub, kaggle CLI) falla con error 403,
  independientemente del método usado. Una vez aceptadas las reglas:
  ```
  python scripts/download_home_credit.py
  ```
  Este script descarga únicamente `application_train.csv` (no el resto de tablas
  auxiliares, ver sección de decisiones de diseño) y lo copia a
  `data/home-credit-default/application_train.csv`.

## Variables excluidas por decisión de diseño (fairness)

`personal_status` y `foreign_worker` (German Credit) y `SEX` (Credit Card) se
excluyen deliberadamente del conjunto de features del modelo. Motivo documentado
en `src/data_loader.py` y `src/data_loader_credit_card.py`: `personal_status`
codifica sexo + estado civil en una sola columna, y `foreign_worker` tiene un
error de codificación conocido en la documentación oficial de UCI (ver Ferrando
et al., "Algorithmic Fairness Datasets: the Story so Far", 2022, arXiv:2202.01711).
Alineado con el Art. 10 del EU AI Act (gobernanza de datos en sistemas de alto riesgo).

Ambos loaders exponen `audit_data_quality()` / `SENSITIVE_FEATURES_EXCLUDED` para
que la exclusión y la calidad de datos (duplicados, nulos, rangos imposibles)
queden documentadas y sean auditables desde el propio código, no solo en la memoria.

## Métricas de evaluación

`evaluate.py` reporta AUC-ROC, Gini, KS, Brier score (calibración) y coste
esperado según la matriz de coste oficial de UCI (falso negativo = 5, falso
positivo = 1). El coste se aplica al **umbral de decisión**, nunca al
entrenamiento — así las probabilidades permanecen calibradas para que
`sum(shap_values) + base_value == proba_default` (local accuracy) siga siendo válido.

## Orden de ejecución

```bash
# --- German Credit (pipeline principal) ---
python src/train_model.py
python src/prepare_background.py
python src/graph.py

# --- Credit Card (validacion cruzada de metricas) ---
python src/train_model_credit_card.py

# --- Comparativa de metricas (funciona con 2 o 3 datasets entrenados) ---
python src/evaluate_cross_dataset.py

# --- Tests ---
pytest tests/ -v

# --- Home Credit: DIFERIDO. Descomentar/ejecutar mas adelante si procede ---
# python scripts/download_home_credit.py
# python src/train_model_home_credit.py    # tarda mas: ~300k filas
```

## Estructura

```
tfm-scoring-crediticio/
├── data/german-credit-data/
│   └── german.data            # dataset original, sin modificar
├── models/                    # modelo entrenado + background SHAP (gitignored)
├── notebooks/                 # exploración inicial, no productivo
├── src/
│   ├── data_loader.py               # German Credit
│   ├── data_loader_credit_card.py   # Default Credit Card
│   ├── data_loader_home_credit.py   # Home Credit Default
│   ├── dataset_utils.py             # split_data() generico compartido
│   ├── train_model.py               # Entrenamiento German Credit
│   ├── train_model_credit_card.py   # Entrenamiento Credit Card
│   ├── train_model_home_credit.py   # Entrenamiento Home Credit (a escala)
│   ├── evaluate.py                  # Métricas AUC/KS/Gini (genérico)
│   ├── evaluate_cross_dataset.py    # Comparativa entre los 3 datasets
│   ├── scoring.py                   # Nodo 1 — conversión proba -> score
│   ├── prepare_background.py        # Background para SHAP (German Credit)
│   ├── graph_state.py               # Estado compartido del grafo
│   ├── nodes.py                     # Nodo 1 y Nodo 2 (LangGraph, German Credit)
│   └── graph.py                     # Ensamblado del StateGraph
└── tests/
    └── test_graph.py          # Test de integración: local accuracy
```

## Referencia académica

Guía completa de diseño e implementación: ver `Guia_Academica_TFM_Scoring_Crediticio.pdf`
y `Guia_Inicio_Implementacion_Nodo1_Nodo2_LangGraph.pdf`.
