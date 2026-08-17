"""
Selecciona (o carga, si ya existen) las 20 instancias de test que se usaran
en TODA la experimentacion H x B. Deben ser siempre las mismas -- por eso
se persisten a disco la primera vez y las ejecuciones siguientes solo leen
el fichero, nunca vuelven a muestrear.

Reutiliza la logica de diversidad ya validada en generate_shap_examples.py
(cobertura de percentiles de riesgo + feature dominante distinto), extendida
de n=5 a n=20 sin cambios de fondo.
"""
import json
import os

import joblib

from data_loader import load_german_credit, split_data
from generate_shap_examples import seleccionar_instancias_diversas
from narrativas_hand_written import NARRATIVAS_HAND_WRITTEN
from experiment_config import N_TEST_INSTANCES, TEST_INSTANCES_SEED, TEST_INSTANCES_FILE

# CRITICO: las instancias que sirvieron de base para las 5 narrativas
# hand-written NUNCA deben aparecer en el conjunto de test evaluado. Si
# una instancia esta en ambos sitios, para configuraciones con H>=1 el
# NARRATOR recibe la respuesta correcta de esa instancia exacta como demo
# y luego se le pide que la narre -- no generaliza, la copia. Contaminacion
# de train/test clasica, detectada tras ver que la instancia 45 (fuente de
# la narrativa H nº1) obtenia valores SHAP identicos a su propio demo en
# la configuracion H1_B0.
INDICES_EXCLUIDOS_POR_H = {item["indice_test"] for item in NARRATIVAS_HAND_WRITTEN}


def cargar_o_seleccionar_instancias(path: str = TEST_INSTANCES_FILE) -> list:
    """
    Devuelve una lista de dicts, uno por instancia:
        {"indice_test": int, "client_data": dict}

    Si el fichero ya existe, se carga tal cual (garantiza reproducibilidad
    exacta entre sesiones). Si no existe, se seleccionan N_TEST_INSTANCES
    instancias diversas -- EXCLUYENDO las 5 fuente de las narrativas H --
    y se persisten.
    """
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            instancias = json.load(fh)
        if len(instancias) != N_TEST_INSTANCES:
            raise RuntimeError(
                f"{path} contiene {len(instancias)} instancias, se esperaban "
                f"{N_TEST_INSTANCES}. No se regenera automaticamente para no "
                f"romper la reproducibilidad -- borra el fichero a mano si "
                f"realmente quieres reseleccionar."
            )
        return instancias

    model = joblib.load("models/xgb_scoring_pipeline.joblib")
    df = load_german_credit("data/german-credit-data/german.data")
    _, X_test, _, _ = split_data(df)

    # Reset + exclusion ANTES de la seleccion diversa: las instancias fuente
    # de las narrativas H se eliminan por completo del pool candidato, para
    # que sea imposible que la seleccion diversa las vuelva a escoger.
    X_test = X_test.reset_index(drop=True)
    mascara_excluidas = X_test.index.isin(INDICES_EXCLUIDOS_POR_H)
    n_excluidas = int(mascara_excluidas.sum())
    X_test_sin_H = X_test[~mascara_excluidas].reset_index(drop=True)
    print(f"Excluidas {n_excluidas} instancias (fuente de narrativas H) del "
          f"pool de seleccion del test set: {sorted(INDICES_EXCLUIDOS_POR_H)}")

    seleccion = seleccionar_instancias_diversas(
        X_test_sin_H, model, n=N_TEST_INSTANCES, seed=TEST_INSTANCES_SEED
    )

    instancias = [
        {"indice_test": idx, "client_data": estado["client_data"]}
        for idx, estado in seleccion
    ]

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(instancias, fh, ensure_ascii=False, indent=2)

    print(f"Seleccionadas y guardadas {len(instancias)} instancias en {path}. "
          f"A partir de ahora todas las ejecuciones reutilizaran este fichero.")
    return instancias


def evaluar_estado_completo(client_data: dict) -> dict:
    """
    Ejecuta node_scoring + node_explainability sobre un client_data ya
    seleccionado, para obtener score/proba_default/aprobado/nivel_riesgo/
    top_features -- todo lo que necesitan el NARRATOR y el GRADER.

    Reutiliza _evaluar_instancia de generate_shap_examples.py adaptandolo
    a recibir client_data directamente en vez de un indice de X_test.
    """
    from nodes import node_scoring, node_explainability
    state = {"client_data": client_data}
    state.update(node_scoring(state))
    state.update(node_explainability(state))
    return state


if __name__ == "__main__":
    instancias = cargar_o_seleccionar_instancias()
    print(f"{len(instancias)} instancias listas en {TEST_INSTANCES_FILE}")
    for inst in instancias[:3]:
        print(f"  indice_test={inst['indice_test']}")
