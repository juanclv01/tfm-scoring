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

## Dataset

German Credit Dataset (UCI, CC BY 4.0). Coloca el fichero en:
`data/german-credit-data/german.data`

Fuente original: https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data

Nota: el zip original de UCI trae 4 ficheros (`german.data`, `german.data-numeric`,
`german.doc`, etc.). Solo `german.data` es necesario; el `.gitignore` ya excluye
el resto para no versionar ficheros innecesarios.

## Orden de ejecución

```bash
# 1. Entrenar y ajustar hiperparámetros del Nodo 1
python src/train_model.py

# 2. Evaluar métricas (AUC-ROC, Gini, KS)
python src/evaluate.py

# 3. Generar el dataset de background para SHAP (necesario antes del Nodo 2)
python src/prepare_background.py

# 4. Ejecutar el grafo completo (Nodo 1 -> Nodo 2) end-to-end
python src/graph.py

# 5. Correr los tests (incluye el chequeo de local accuracy)
pytest tests/ -v
```

## Estructura

```
tfm-scoring-crediticio/
├── data/german-credit-data/
│   └── german.data            # dataset original, sin modificar
├── models/                    # modelo entrenado + background SHAP (gitignored)
├── notebooks/                 # exploración inicial, no productivo
├── src/
│   ├── data_loader.py         # Nodo 1 — carga y preprocesado
│   ├── train_model.py         # Nodo 1 — entrenamiento y tuning
│   ├── evaluate.py            # Nodo 1 — métricas AUC/KS/Gini
│   ├── scoring.py             # Nodo 1 — conversión proba -> score
│   ├── prepare_background.py  # Genera el background dataset para SHAP
│   ├── graph_state.py         # Estado compartido del grafo
│   ├── nodes.py                # Nodo 1 y Nodo 2 como funciones LangGraph
│   └── graph.py                # Ensamblado del StateGraph
└── tests/
    └── test_graph.py          # Test de integración: local accuracy
```

## Referencia académica

Guía completa de diseño e implementación: ver `Guia_Academica_TFM_Scoring_Crediticio.pdf`
y `Guia_Inicio_Implementacion_Nodo1_Nodo2_LangGraph.pdf`.
