"""
Calcula la tasa de impago EMPIRICA (% de target="malo") por categoria,
directamente sobre los datos etiquetados -- SIN pasar por el modelo
XGBoost ni por SHAP en ningun momento.

Motivacion: analyze_categorical_pattern.py y analyze_numeric_pattern.py
muestran que ciertas variables (checking_status, credit_history,
installment_rate) tienen un efecto SHAP fuertemente sesgado y casi
determinista por categoria. Antes de asumir que esto refleja un sesgo de
seleccion INHERENTE AL DATASET (hipotesis de reject inference, ver
conversacion de diseño), hay que descartar la hipotesis alternativa: que
sea un artefacto introducido por el propio pipeline de ML (XGBoost
sobreajustando con solo ~800 filas de train, o una configuracion
incorrecta de TreeExplainer).

La forma de diferenciar ambas hipotesis es comparar el patron observado
via SHAP contra la tasa de impago CRUDA por categoria, calculada
directamente sobre las etiquetas del dataset (sin modelo de por medio).
Si el patron persiste aqui -- p.ej. clientes 'sin cuenta corriente'
tienen una tasa de impago real menor, en los datos tal cual fueron
etiquetados en 1973-75 -- entonces el sesgo esta grabado en el dataset
mismo, no puede ser un artefacto de XGBoost ni de SHAP (ninguno de los
dos interviene en este calculo). Es el mismo metodo que usa la propia
literatura de reject inference para diagnosticar este problema.

Uso:
    python verify_raw_bad_rate.py --feature checking_status
    python verify_raw_bad_rate.py --feature credit_history
    python verify_raw_bad_rate.py --feature installment_rate
"""
import argparse

import pandas as pd

from data_loader import load_german_credit, CATEGORICAL_FEATURES, NUMERIC_FEATURES, decode_feature_value


def tasa_impago_por_categoria(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """
    df['target']: 1 = "malo" (impago), 0 = "bueno" -- ver data_loader.py.
    Se usa el DATASET COMPLETO (no solo el test set) para maximizar n y
    porque aqui no hay riesgo de fuga: no se esta entrenando ni evaluando
    ningun modelo, solo describiendo la variable objetivo tal cual viene
    etiquetada.
    """
    resumen = df.groupby(feature)["target"].agg(
        n="count", tasa_impago_pct=lambda s: s.mean() * 100,
    ).reset_index()

    if feature in CATEGORICAL_FEATURES:
        resumen["descripcion_es"] = resumen[feature].apply(
            lambda c: decode_feature_value(feature, c)
        )
    else:
        resumen["descripcion_es"] = resumen[feature].astype(str)

    return resumen.sort_values(feature)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", type=str, required=True,
                         help="variable a analizar (categorica o numerica)")
    args = parser.parse_args()

    if args.feature not in CATEGORICAL_FEATURES and args.feature not in NUMERIC_FEATURES:
        raise ValueError(
            f"'{args.feature}' no es una feature activa del modelo. "
            f"Categoricas: {CATEGORICAL_FEATURES}. Numericas: {NUMERIC_FEATURES}."
        )

    df = load_german_credit("data/german-credit-data/german.data")

    resumen = tasa_impago_por_categoria(df, args.feature)

    print(f"\nTasa de impago EMPIRICA por categoria -- {args.feature} "
          f"(n={len(df)} filas, dataset completo, SIN modelo ni SHAP)\n")
    print(resumen.to_string(index=False))

    print("\nInterpretacion:")
    print("- Si la tasa de impago real (tasa_impago_pct) YA muestra el mismo")
    print("  patron contraintuitivo que el SHAP agregado (p.ej. 'sin cuenta")
    print("  corriente' con MENOR tasa de impago real), el sesgo esta grabado")
    print("  en el dataset -- no puede ser un artefacto de XGBoost ni de SHAP,")
    print("  ninguno de los dos interviene en este calculo.")
    print("- Si la tasa de impago real NO muestra el patron (o lo muestra muy")
    print("  atenuado) y el efecto SHAP si es fuerte, el modelo esta")
    print("  amplificando o inventando una relacion que los datos crudos no")
    print("  sostienen tan claramente -- en ese caso si merece revisarse el")
    print("  modelo (sobreajuste, hiperparametros) antes de documentarlo como")
    print("  limitacion del dataset.")


if __name__ == "__main__":
    main()
