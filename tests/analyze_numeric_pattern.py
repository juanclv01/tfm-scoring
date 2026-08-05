"""
Analiza si la contribucion SHAP de una variable NUMERICA PASSTHROUGH
(ordinal, como installment_rate, residence_since, existing_credits,
num_dependents) es un patron sistematico en todo el test set, o ruido de
un puñado de instancias -- mismo objetivo que analyze_categorical_pattern.py,
pero para variables que el preprocesador deja como passthrough (una sola
columna transformada 'num__{feature}'), no OneHotEncoded.

NO uses analyze_categorical_pattern.py para estas variables: busca columnas
'cat__{feature}_*' y lanzaria ValueError al no encontrar ninguna -- una
feature passthrough no tiene dummies que agregar, es ya una unica columna.

Motivacion inmediata: 'installment_rate' (ya corregido en su decodificacion
a tramos, ver data_loader.INSTALLMENT_RATE_LABELS) muestra un patron de
signo que no es monotono con el codigo (1->+9.6, 2->+7.5, 4->-6.1 en las
narrativas de ejemplo) -- contraintuitivo, y sin la n suficiente en 3
instancias para saber si es sistematico o ruido de esas instancias
concretas. Este script responde esa pregunta sobre el test set completo.

Uso:
    python analyze_numeric_pattern.py --feature installment_rate
    python analyze_numeric_pattern.py --feature residence_since
    python analyze_numeric_pattern.py --feature existing_credits
    python analyze_numeric_pattern.py --feature num_dependents
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
    NUMERIC_FEATURES,
    FEATURE_NAME_LABELS_ES,
    INSTALLMENT_RATE_LABELS,
)
from scoring import SCORE_MAX, SCORE_MIN


def _etiqueta_valor(feature: str, valor) -> str:
    """
    Etiqueta legible para un valor concreto, si existe un mapeo conocido
    y VERIFICADO (por ahora solo installment_rate -- ver conversacion de
    diseño sobre residence_since/existing_credits, cuyos brackets NO
    estan confirmados por una fuente fiable; para esas se muestra el
    codigo crudo tal cual, sin inventar una etiqueta).
    """
    if feature == "installment_rate":
        return INSTALLMENT_RATE_LABELS.get(int(valor), str(valor))
    return str(valor)


def compute_shap_for_numeric_feature(feature: str, model, X_test: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve un DataFrame con una fila por instancia del test set:
    'codigo_real' (valor real de la columna passthrough) y 'shap_score'
    (contribucion SHAP de esa UNICA columna transformada, ya en PUNTOS DE
    SCORE) -- sin necesidad de agregar nada, a diferencia de las
    categoricas OHE en analyze_categorical_pattern.py.
    """
    if feature not in NUMERIC_FEATURES:
        raise ValueError(
            f"'{feature}' no es una variable numerica passthrough activa "
            f"en el modelo. Para variables categoricas (OneHotEncoder), "
            f"usa analyze_categorical_pattern.py en su lugar."
        )

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
    nombre_transformado = f"num__{feature}"
    if nombre_transformado not in feature_names:
        raise ValueError(
            f"No se encontro la columna '{nombre_transformado}' entre las "
            f"features transformadas -- revisa NUMERIC_FEATURES en data_loader.py."
        )
    indice = feature_names.index(nombre_transformado)

    shap_proba = shap_values_proba[:, indice]
    shap_score = -shap_proba * (SCORE_MAX - SCORE_MIN)

    return pd.DataFrame({
        "codigo_real": X_test[feature].reset_index(drop=True),
        "shap_score": shap_score,
    })


def resumen_por_valor(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """
    Estadisticos por valor -- mismo criterio que
    analyze_categorical_pattern.resumen_por_categoria(). Se ordena por
    'codigo_real' (no por 'media') a proposito: para una variable
    ORDINAL, ver la tabla en el orden natural del codigo (1,2,3,4) es lo
    que permite juzgar a simple vista si la relacion es monotona o no.
    """
    agrupado = df.groupby("codigo_real")["shap_score"]
    resumen = agrupado.agg(
        n="count", media="mean", mediana="median", std="std",
        minimo="min", maximo="max",
    ).reset_index()
    resumen["pct_positivo"] = agrupado.apply(lambda s: (s > 0).mean() * 100).values
    resumen["descripcion_es"] = resumen["codigo_real"].apply(
        lambda v: _etiqueta_valor(feature, v)
    )
    return resumen.sort_values("codigo_real")


def graficar_boxplot(df: pd.DataFrame, feature: str, out_path: str):
    valores = sorted(df["codigo_real"].unique())
    datos = [df[df["codigo_real"] == v]["shap_score"].values for v in valores]
    etiquetas = [_etiqueta_valor(feature, v)[:35] for v in valores]

    fig, ax = plt.subplots(figsize=(10, 0.8 * len(valores) + 2))
    # CORRECTED: matplotlib >= 3.11 deprecia 'vert' en favor de
    # 'orientation'; matplotlib 3.9-3.10 acepta 'vert' pero ya no
    # 'labels' (solo 'tick_labels'); matplotlib < 3.9 solo acepta
    # 'labels'. Se prueban las tres combinaciones de mas reciente a mas
    # antigua para no romper segun la version instalada (requirements.txt
    # solo fija matplotlib>=3.8, un rango amplio).
    try:
        ax.boxplot(datos, tick_labels=etiquetas, orientation="horizontal")
    except TypeError:
        try:
            ax.boxplot(datos, tick_labels=etiquetas, vert=False)
        except TypeError:
            ax.boxplot(datos, labels=etiquetas, vert=False)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Contribucion SHAP (puntos de score)")
    nombre_es = FEATURE_NAME_LABELS_ES.get(feature, feature)
    ax.set_title(f"Distribucion de SHAP por valor -- {nombre_es}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nGrafico guardado en {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", type=str, required=True,
                         help="variable numerica passthrough a analizar (p.ej. installment_rate)")
    parser.add_argument("--out", type=str, default=None,
                         help="ruta del boxplot (default: <feature>_shap_boxplot.png)")
    args = parser.parse_args()

    model = joblib.load("models/xgb_scoring_pipeline.joblib")
    df_full = load_german_credit("data/german-credit-data/german.data")
    _, X_test, _, _ = split_data(df_full)

    df = compute_shap_for_numeric_feature(args.feature, model, X_test)
    resumen = resumen_por_valor(df, args.feature)

    print(f"\nResumen de SHAP por valor -- {args.feature} (n={len(X_test)} en test set)\n")
    print(resumen.to_string(index=False))

    out_path = args.out or f"{args.feature}_shap_boxplot.png"
    graficar_boxplot(df, args.feature, out_path)

    print("\nInterpretacion sugerida (mismo criterio que analyze_categorical_pattern.py):")
    print("- pct_positivo cercano a 0% o 100% -> patron sistematico de ese valor.")
    print("- pct_positivo cercano a 50% -> ruido de interaccion; evitar generalizar")
    print("  en la narrativa hand-written a partir de una sola instancia asi.")
    print("- Especifico de variables ORDINALES: revisa ademas si 'media' cambia")
    print("  de forma MONOTONA con el codigo (1->2->3->4) o si salta sin orden --")
    print("  lo segundo (no monotono) es mas dificil de justificar como relacion")
    print("  causal simple y merece mencion explicita como limitacion si aparece.")


if __name__ == "__main__":
    main()
