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

| Dataset | Rol | Fuente | Ruta |
|---|---|---|---|
| German Credit | Prototipo inicial + modelo del pipeline en vivo | UCI | `data/german-credit-data/german.data` |
| Default Credit Card | Validación cruzada de métricas | UCI/Kaggle | `data/credit-card-default/UCI_Credit_Card.csv` |
| Home Credit Default | Validación final, prueba de escala | Kaggle | `data/home-credit-default/application_train.csv` |

**Importante:** el grafo en vivo (`graph.py`, `nodes.py`) sigue alimentado únicamente
por German Credit. Los otros dos datasets son validaciones offline, no sustituyen
el modelo del pipeline de demo.

**Descarga:**
- German Credit: ver `data/LEEME.txt`.
- Default Credit Card: Kaggle — `uciml/default-of-credit-card-clients-dataset`. ~2-3 MB, sí se versiona en git.
- Home Credit Default: Kaggle — competición `home-credit-default-risk`, fichero `application_train.csv`.
  **No se versiona en git** (~150-160 MB, supera el límite de 100 MB/fichero de GitHub).
  Descárgalo con la Kaggle API: `kaggle competitions download -c home-credit-default-risk -f application_train.csv`
  y colócalo manualmente en `data/home-credit-default/`.

## Datasets utilizados

| Dataset | Rol | Fuente | Ruta |
|---|---|---|---|
| German Credit | Prototipo inicial + modelo del pipeline en vivo | UCI | `data/german-credit-data/german.data` |
| Default Credit Card | Validación cruzada de métricas | UCI/Kaggle | `data/credit-card-default/UCI_Credit_Card.csv` |
| Home Credit Default | Validación final, prueba de escala | Kaggle | `data/home-credit-default/application_train.csv` |

**Importante:** el grafo en vivo (`graph.py`, `nodes.py`) sigue alimentado únicamente
por German Credit. Los otros dos datasets son validaciones offline, no sustituyen
el modelo del pipeline de demo.

## Orden de ejecución

```bash
# --- German Credit (pipeline principal) ---
python src/train_model.py
python src/prepare_background.py
python src/graph.py

# --- Datasets de validación adicional ---
python src/train_model_credit_card.py
python src/train_model_home_credit.py    # tarda mas: ~300k filas

# --- Comparativa de metricas entre los tres ---
python src/evaluate_cross_dataset.py

# --- Tests ---
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
