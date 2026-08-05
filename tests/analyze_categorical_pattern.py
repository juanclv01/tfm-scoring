"""
Analiza si la contribucion SHAP de una variable categorica es un PATRON
SISTEMATICO en todo el test set, o un efecto puntual de un puñado de
instancias (ruido de interaccion) -- distincion que 5 narrativas de
ejemplo no bastan para hacer con confianza.

Motivacion: en la revision de las narrativas de ejemplo se observo que
'credit_history' mostraba signos opuestos para categorias intuitivamente
cercanas (A34 'cuenta critica' con SHAP positivo en una instancia, A32
'pagos puntuales' con SHAP negativo en otra). Para 'checking_status' este
mismo tipo de patron contraintuitivo SI esta confirmado como real por
literatura externa (no un bug del pipeline) -- ver conversacion de diseño.
Para 'credit_history' solo se disponia de 2 puntos, insuficiente para
diferenciar "patron sistematico" de "coincidencia".

Reutiliza el preprocesador y el TreeExplainer con la misma configuracion
que node_explainability() (model_output='probability',
feature_perturbation='interventional'), pero calcula SHAP para TODO
X_test en una sola llamada -- no tiene sentido invocar
node_scoring()/node_explainability() fila a fila para un analisis
exploratorio de cientos de instancias (ver generate_shap_examples.py
para el patron equivalente ya usado alli, pero orientado a seleccionar
5 instancias, no a analizar la totalidad).

La agregacion (sumar las contribuciones de todas las dummies OHE de una
misma variable) usa el mismo criterio que data_loader.aggregate_shap_by_feature(),
pero vectorizado sobre matrices en vez de diccionarios por instancia.

Requiere haber ejecutado train_model.py y prepare_background.py antes.

Uso:
    python analyze_categorical_pattern.py --feature credit_history
    python analyze_categorical_pattern.py --feature checking_status
"""
import argparse

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from data_loader import (
    load_german_credit,
    split_data,
    CATEGORICAL_FEATURES,
    FEATURE_NAME_LABELS_ES,
    decode_feature_value,
)
from scoring import SCORE_MAX, SCORE_MIN


def compute_aggregated_shap_for_feature(feature: str, model, X_test: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve un DataFrame con una fila por instancia del test set:
    'codigo_real' (categoria real del cliente para 'feature', tal cual
    aparece en los datos crudos) y 'shap_score_agregado' (suma de todas
    las dummies OHE de esa variable, ya en PUNTOS DE SCORE -- mismo
    signo/escala que usa el resto del pipeline desde node_explainability()).
    """
    if feature not in CATEGORICAL_FEATURES:
        raise ValueError(f"'{feature}' no es una variable categorica activa en el modelo.")

    preprocessor = model.named_steps["preprocessor"]
    xgb_model = model.named_steps["model"]
    X_transformed = preprocessor.transform(X_test)

    background = joblib.load("models/background_sample.joblib")
    explainer = shap.TreeExplainer(
        xgb_model, data=background,
        feature_perturbation="interventional", model_output="probability",
    )
    shap_values_proba = np.asarray(explainer.shap_values(X_transformed))

    feature_names = preprocessor.get_feature_names_out().tolist()
    prefijo = f"cat__{feature}_"
    indices_feature = [i for i, nombre in enumerate(feature_names) if nombre.startswith(prefijo)]
    if not indices_feature:
        raise ValueError(f"No se encontraron columnas dummy para '{feature}'.")

    # Suma en espacio de proba_default (local accuracy se preserva: es una
    # suma parcial, no una alteracion de valores), luego conversion a
    # puntos de score -- mismo orden logico que node_explainability().
    shap_proba_agregado = shap_values_proba[:, indices_feature].sum(axis=1)
    shap_score_agregado = -shap_proba_agregado * (SCORE_MAX - SCORE_MIN)

    return pd.DataFrame({
        "codigo_real": X_test[feature].reset_index(drop=True),
        "shap_score_agregado": shap_score_agregado,
    })


def resumen_por_categoria(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """
    Estadisticos por categoria real. 'pct_positivo' es la señal clave:
    cercano a 0% o 100% dentro de una categoria indica que el signo es
    un rasgo estable de ESA categoria; cercano a 50% indica que el signo
    depende mas de la interaccion con el resto del perfil del cliente
    que de la categoria en si (es decir, mas cercano a ruido).
    """
    agrupado = df.groupby("codigo_real")["shap_score_agregado"]
    resumen = agrupado.agg(
        n="count", media="mean", mediana="median", std="std",
        minimo="min", maximo="max",
    ).reset_index()
    resumen["pct_positivo"] = agrupado.apply(lambda s: (s > 0).mean() * 100).values
    resumen["descripcion_es"] = resumen["codigo_real"].apply(
        lambda c: decode_feature_value(feature, c)
    )
    return resumen.sort_values("media", ascending=False)


def graficar_boxplot(df: pd.DataFrame, feature: str, out_path: str):
    categorias = sorted(df["codigo_real"].unique())
    datos = [df[df["codigo_real"] == c]["shap_score_agregado"].values for c in categorias]
    etiquetas = [decode_feature_value(feature, c)[:35] for c in categorias]

    fig, ax = plt.subplots(figsize=(10, 0.8 * len(categorias) + 2))
    # CORRECTED: matplotlib >= 3.11 deprecia tambien 'vert' (ademas del
    # 'labels' ya corregido antes) en favor de 'orientation'. Mismo
    # fallback en cascada que analyze_numeric_pattern.py, para cubrir
    # matplotlib < 3.9, 3.9-3.10, y >= 3.11 sin fijar una version exacta
    # en requirements.txt (que solo declara matplotlib>=3.8).
    try:
        ax.boxplot(datos, tick_labels=etiquetas, orientation="horizontal")
    except TypeError:
        try:
            ax.boxplot(datos, tick_labels=etiquetas, vert=False)
        except TypeError:
            ax.boxplot(datos, labels=etiquetas, vert=False)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Contribucion SHAP agregada (puntos de score)")
    nombre_es = FEATURE_NAME_LABELS_ES.get(feature, feature)
    ax.set_title(f"Distribucion de SHAP por categoria -- {nombre_es}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nGrafico guardado en {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", type=str, required=True,
                         help="variable categorica a analizar (p.ej. credit_history)")
    parser.add_argument("--out", type=str, default=None,
                         help="ruta del boxplot (default: <feature>_shap_boxplot.png)")
    args = parser.parse_args()

    model = joblib.load("models/xgb_scoring_pipeline.joblib")
    df_full = load_german_credit("data/german-credit-data/german.data")
    _, X_test, _, _ = split_data(df_full)

    df = compute_aggregated_shap_for_feature(args.feature, model, X_test)
    resumen = resumen_por_categoria(df, args.feature)

    print(f"\nResumen de SHAP agregado por categoria -- {args.feature} (n={len(X_test)} en test set)\n")
    print(resumen.to_string(index=False))

    out_path = args.out or f"{args.feature}_shap_boxplot.png"
    graficar_boxplot(df, args.feature, out_path)

    print("\nInterpretacion sugerida (no automatica, revisar la tabla con criterio):")
    print("- pct_positivo cercano a 0% o 100% dentro de una categoria -> patron")
    print("  sistematico de ESA categoria. Documentar como limitacion del dataset")
    print("  (citando literatura), no como bug del pipeline.")
    print("- pct_positivo cercano a 50% -> el signo observado en una instancia")
    print("  puntual depende mas de la interaccion con el resto del perfil del")
    print("  cliente que de la categoria en si -- mas cercano a ruido que a un")
    print("  patron atribuible a la categoria. Evitar generalizar en la narrativa")
    print("  hand-written a partir de una sola instancia asi.")


if __name__ == "__main__":
    main()
