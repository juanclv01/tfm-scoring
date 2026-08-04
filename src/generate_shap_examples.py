"""
Genera N conjuntos de valores SHAP (por defecto N=5), en formato NARRATOR
-- (feature_name_es, feature_value_es, shap_value) --, a partir del conjunto
de test, para servir de base a las narrativas hand-written (H) del NARRATOR.

Reutiliza node_scoring() y node_explainability() de nodes.py en vez de
reimplementar el calculo de score/SHAP: evita logica duplicada, y garantiza
que los ejemplares se generan exactamente con el mismo pipeline (mismo
model_output='probability', mismo top-5 por |SHAP|, mismo build_narrator_tuple)
que usara luego el NARRATOR en produccion.

Criterio de diversidad (para reducir sesgo, tal como pide el TFM):
    (a) Cobertura del rango de riesgo: se parte de percentiles equiespaciados
        de proba_default sobre todo el test set (no solo casos extremos).
    (b) Feature dominante distinto: si dos candidatos comparten el mismo
        feature con |SHAP| maximo, se sustituye uno de ellos por la instancia
        mas cercana (en el ranking de proba_default) que tenga un feature
        dominante diferente, dentro de un radio de busqueda acotado.

Requiere haber ejecutado train_model.py y prepare_background.py antes
(el modelo y el background de SHAP deben existir en models/).

Uso:
    python generate_shap_examples.py
    python generate_shap_examples.py --n 5 --out shap_examples.json --seed 42
"""
import argparse
import json

import joblib
import numpy as np
import pandas as pd

from nodes import node_scoring, node_explainability
from data_loader import load_german_credit, split_data

RADIO_BUSQUEDA = 15  # max. instancias vecinas a probar si hay colision de feature dominante


def _proba_default_test(model, X_test: pd.DataFrame) -> np.ndarray:
    """Probabilidad de impago para todo el test set, en una sola pasada
    por el pipeline (evita llamar a node_scoring fila a fila, que es O(n)
    invocaciones de predict_proba con un solo registro cada vez)."""
    return model.predict_proba(X_test)[:, 1]


def _feature_dominante(top_features: list) -> str:
    """feature_raw del feature con |shap_value| maximo dentro del top-5
    ya ordenado (top_features[0] es, por construccion en nodes.py, el de
    mayor |SHAP|)."""
    return top_features[0]["feature_raw"]


def _evaluar_instancia(idx: int, X_test: pd.DataFrame) -> dict:
    """Ejecuta node_scoring + node_explainability para una fila del test
    set, reutilizando el contrato de PipelineState (dict) tal cual lo
    consumen ambos nodos."""
    client_data = X_test.iloc[idx].to_dict()
    state = {"client_data": client_data}
    state.update(node_scoring(state))
    state.update(node_explainability(state))
    return state


def seleccionar_instancias_diversas(X_test: pd.DataFrame, model, n: int, seed: int) -> list:
    """
    Devuelve una lista de n indices (posicionales, sobre X_test.reset_index)
    diversos en riesgo predicho y en feature dominante.
    """
    X_test = X_test.reset_index(drop=True)
    proba = _proba_default_test(model, X_test)

    orden = np.argsort(proba)  # indices ordenados de menor a mayor riesgo
    n_total = len(orden)

    # Percentiles equiespaciados (0, 100/(n-1), ..., 100) sobre el orden de riesgo
    posiciones_percentil = np.linspace(0, n_total - 1, n).round().astype(int)
    candidatos = [int(orden[p]) for p in posiciones_percentil]

    seleccion = []
    features_dominantes_usados = set()

    for i, idx in enumerate(candidatos):
        estado = _evaluar_instancia(idx, X_test)
        feat_dom = _feature_dominante(estado["top_features"])

        if feat_dom not in features_dominantes_usados:
            seleccion.append((idx, estado))
            features_dominantes_usados.add(feat_dom)
            continue

        # Colision: buscar el vecino mas cercano en el ranking de riesgo
        # (a ambos lados de la posicion original) con feature dominante distinto.
        pos_original = posiciones_percentil[i]
        candidato_alternativo = None
        for radio in range(1, RADIO_BUSQUEDA + 1):
            for pos_vecina in (pos_original - radio, pos_original + radio):
                if not (0 <= pos_vecina < n_total):
                    continue
                idx_vecino = int(orden[pos_vecina])
                if idx_vecino in [s[0] for s in seleccion]:
                    continue
                estado_vecino = _evaluar_instancia(idx_vecino, X_test)
                feat_vecino = _feature_dominante(estado_vecino["top_features"])
                if feat_vecino not in features_dominantes_usados:
                    candidato_alternativo = (idx_vecino, estado_vecino)
                    break
            if candidato_alternativo:
                break

        if candidato_alternativo:
            seleccion.append(candidato_alternativo)
            features_dominantes_usados.add(
                _feature_dominante(candidato_alternativo[1]["top_features"])
            )
        else:
            # No se encontro alternativa en el radio de busqueda: se conserva
            # el candidato original para no perder cobertura del percentil.
            seleccion.append((idx, estado))
            features_dominantes_usados.add(feat_dom)

    return seleccion


def formatear_tuplas_narrator(top_features: list) -> list:
    # shap_value ya viene en PUNTOS DE SCORE (convertido por
    # node_explainability, no en espacio de probabilidad): positivo sube
    # el score, negativo lo baja. Formato "+.1f pts" en vez de ".4f",
    # consistente con data_loader.py y graph.py.
    return [
        f"({f['feature_name']}, {f['feature_value']}, {f['shap_value']:+.1f} pts)"
        for f in top_features
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5,
                         help="numero de conjuntos SHAP a generar (default: 5)")
    parser.add_argument("--out", type=str, default="shap_examples.json",
                         help="fichero de salida (default: shap_examples.json)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model = joblib.load("models/xgb_scoring_pipeline.joblib")
    df = load_german_credit("data/german-credit-data/german.data")
    _, X_test, _, _ = split_data(df)

    seleccion = seleccionar_instancias_diversas(X_test, model, n=args.n, seed=args.seed)

    resultados = []
    for i, (idx, estado) in enumerate(seleccion, start=1):
        resultados.append({
            "instancia_id": i,
            "indice_test": idx,
            "client_data": estado["client_data"],
            "score": estado["score"],
            "proba_default": estado["proba_default"],
            "top_features": estado["top_features"],
        })

        print(f"--- Instancia {i} (indice testset: {idx}) ---")
        print(f"Score: {estado['score']} / 1000  |  "
              f"Probabilidad de impago: {estado['proba_default']:.2%}")
        print("Top 5 factores SHAP (formato NARRATOR):")
        for linea in formatear_tuplas_narrator(estado["top_features"]):
            print(f"  {linea}")
        print()

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(resultados, fh, ensure_ascii=False, indent=2)

    print(f"Guardado en {args.out}. Recuerda: shap_value esta en PUNTOS DE "
          f"SCORE (escala 0-1000), no en probabilidad -- shap_value > 0 SUBE "
          f"el score; shap_value < 0 lo BAJA. Misma convencion de signo del "
          f"prompt del NARRATOR.")


if __name__ == "__main__":
    main()
