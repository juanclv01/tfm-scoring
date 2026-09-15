"""
Verificacion de estabilidad de las metricas obtenidas en el test set unico
(AUC-ROC y KS) y de que ninguna feature domina de forma desproporcionada.

Con ~200 filas de test (20% de 1.000), una metrica calculada sobre un unico
split tiene un error estandar no despreciable -- este script responde a la
pregunta de si los valores obtenidos son estables o un accidente favorable
de esa particion concreta, mediante:

  1) AUC-ROC y KS calculados sobre los MISMOS folds de una
     RepeatedStratifiedKFold aplicada al 100% del dataset (no solo train),
     con los mismos hiperparametros y el mismo n_estimators que el modelo
     final ya entrenado. Usar los mismos folds para ambas metricas permite
     comparar directamente su variabilidad relativa, no solo sus medias.
  2) Un chequeo rapido de que ninguna feature domina de forma
     desproporcionada la prediccion (que apuntaria a un atajo trivial del
     dataset en vez de una señal genuina y distribuida).

No vuelve a ejecutar la busqueda de hiperparametros (RandomizedSearchCV):
extrae los hiperparametros y el n_estimators ya fijados en
models/xgb_scoring_pipeline.joblib, y los reutiliza en pipelines SIN
entrenar para las nuevas particiones de CV.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from data_loader import load_german_credit, build_preprocessor
from evaluate import ks_statistic

RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 10   # 5x10 = 50 estimaciones por metrica

# Valores de referencia obtenidos sobre el test set original (para comparar)
AUC_REFERENCIA = 0.8018
KS_REFERENCIA = 0.5048


def extraer_hiperparametros(model_path: str = "models/xgb_scoring_pipeline.joblib") -> dict:
    """
    Lee el pipeline ya entrenado y devuelve los hiperparametros del
    XGBClassifier (incluido n_estimators, ya fijado por xgb.cv en
    train_model.py), listos para reconstruir un modelo SIN entrenar.
    """
    pipeline_entrenado = joblib.load(model_path)
    xgb_entrenado = pipeline_entrenado.named_steps["model"]

    claves_relevantes = [
        "n_estimators", "max_depth", "learning_rate", "subsample",
        "colsample_bytree", "min_child_weight", "gamma",
        "reg_alpha", "reg_lambda",
    ]
    params = xgb_entrenado.get_params()
    return {k: params[k] for k in claves_relevantes if params.get(k) is not None}


def build_pipeline_sin_entrenar(hiperparametros: dict) -> Pipeline:
    """Pipeline nuevo (preprocesador + modelo), sin ajustar, con los
    mismos hiperparametros que el modelo final ya entrenado."""
    preprocessor = build_preprocessor()
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **hiperparametros,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def verificar_estabilidad_auc_y_ks(df: pd.DataFrame, hiperparametros: dict) -> dict:
    """
    Bucle manual sobre RepeatedStratifiedKFold: en cada fold se entrena
    un pipeline nuevo (clonado, sin fugas entre folds) y se calculan
    AUC-ROC y KS sobre el mismo conjunto de validacion del fold. Usar
    los mismos folds para ambas metricas (en vez de dos llamadas
    independientes a cross_val_score) permite comparar su dispersion
    de forma directamente pareada.
    """
    X = df.drop(columns=["target"]).reset_index(drop=True)
    y = df["target"].reset_index(drop=True)

    pipeline_base = build_pipeline_sin_entrenar(hiperparametros)
    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE
    )

    aucs, kss = [], []
    for train_idx, val_idx in cv.split(X, y):
        pipeline_fold = clone(pipeline_base)
        pipeline_fold.fit(X.iloc[train_idx], y.iloc[train_idx])

        y_proba = pipeline_fold.predict_proba(X.iloc[val_idx])[:, 1]
        y_val = y.iloc[val_idx]

        aucs.append(roc_auc_score(y_val, y_proba))
        kss.append(ks_statistic(y_val, y_proba))

    aucs, kss = np.array(aucs), np.array(kss)

    return {
        "n_estimaciones": len(aucs),
        "auc_medio": float(aucs.mean()), "auc_std": float(aucs.std()),
        "auc_min": float(aucs.min()), "auc_max": float(aucs.max()),
        "ks_medio": float(kss.mean()), "ks_std": float(kss.std()),
        "ks_min": float(kss.min()), "ks_max": float(kss.max()),
    }


def verificar_dominancia_de_features(model_path: str = "models/xgb_scoring_pipeline.joblib",
                                      umbral_dominancia: float = 0.5) -> dict:
    """
    Comprueba que ninguna feature acapara mas del `umbral_dominancia`
    (por defecto 50%) de la importancia total del modelo -- una senal
    de que el modelo podria estar explotando un atajo trivial del
    dataset en vez de una señal distribuida entre varias variables.
    Usa la importancia nativa de XGBoost (tipo "gain"), no SHAP, porque
    aqui solo interesa una comprobacion rapida a nivel de modelo global,
    no explicaciones por cliente.
    """
    pipeline_entrenado = joblib.load(model_path)
    preprocessor = pipeline_entrenado.named_steps["preprocessor"]
    xgb_entrenado = pipeline_entrenado.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    importancias = xgb_entrenado.feature_importances_

    orden = np.argsort(importancias)[::-1]
    top_5 = [(feature_names[i], float(importancias[i])) for i in orden[:5]]

    importancia_maxima = float(importancias.max())
    feature_dominante = feature_names[importancias.argmax()]

    return {
        "top_5_features": top_5,
        "importancia_relativa_maxima": importancia_maxima,
        "feature_mas_importante": feature_dominante,
        "supera_umbral_dominancia": importancia_maxima > umbral_dominancia,
    }


def _interpretar_estabilidad(nombre_metrica: str, valor_referencia: float,
                              media: float, std: float) -> None:
    """Imprime el veredicto de estabilidad para una metrica dada,
    comparando el valor del test set original contra la media +/- std
    de la validacion cruzada."""
    diferencia = abs(media - valor_referencia)
    if diferencia <= std:
        print(f"-> El {valor_referencia:.4f} del test set esta dentro de 1 "
              f"desviacion tipica de la media de CV: {nombre_metrica} "
              f"razonablemente estable.")
    else:
        print(f"-> El {valor_referencia:.4f} del test set se aleja mas de 1 "
              f"desviacion tipica de la media de CV: {nombre_metrica} del "
              f"test set probablemente optimista. Considera reportar la "
              f"media de esta CV en la memoria en lugar del valor puntual.")


if __name__ == "__main__":
    hiperparametros = extraer_hiperparametros()
    df = load_german_credit("data/german-credit-data/german.data")

    print("=" * 70)
    print(f"1) ESTABILIDAD DE AUC-ROC Y KS (RepeatedStratifiedKFold, "
          f"{N_SPLITS}x{N_REPEATS}={N_SPLITS * N_REPEATS} estimaciones)")
    print("=" * 70)
    print(f"Hiperparametros extraidos del modelo final: {hiperparametros}\n")

    resultado = verificar_estabilidad_auc_y_ks(df, hiperparametros)

    print("-- AUC-ROC --")
    print(f"AUC medio (CV):      {resultado['auc_medio']:.4f}")
    print(f"Desviacion tipica:   {resultado['auc_std']:.4f}")
    print(f"Rango [min, max]:    [{resultado['auc_min']:.4f}, {resultado['auc_max']:.4f}]")
    print(f"AUC del test set original (referencia): {AUC_REFERENCIA:.4f}")
    _interpretar_estabilidad("AUC-ROC", AUC_REFERENCIA, resultado["auc_medio"], resultado["auc_std"])

    print("\n-- KS --")
    print(f"KS medio (CV):       {resultado['ks_medio']:.4f}")
    print(f"Desviacion tipica:   {resultado['ks_std']:.4f}")
    print(f"Rango [min, max]:    [{resultado['ks_min']:.4f}, {resultado['ks_max']:.4f}]")
    print(f"KS del test set original (referencia): {KS_REFERENCIA:.4f}")
    _interpretar_estabilidad("KS", KS_REFERENCIA, resultado["ks_medio"], resultado["ks_std"])

    # Coeficiente de variacion (std/media): compara la dispersion RELATIVA
    # de ambas metricas entre si, ya que KS y AUC viven en escalas distintas
    # y sus desviaciones tipicas absolutas no son directamente comparables.
    cv_auc = resultado["auc_std"] / resultado["auc_medio"]
    cv_ks = resultado["ks_std"] / resultado["ks_medio"]
    print(f"\nCoeficiente de variacion -- AUC: {cv_auc:.2%}, KS: {cv_ks:.2%}")
    if cv_ks > cv_auc:
        print("-> El KS es relativamente mas variable entre folds que el "
              "AUC-ROC (esperable: el KS depende de un unico umbral optimo, "
              "mientras que el AUC promedia sobre todos los umbrales).")

    print()
    print("=" * 70)
    print("2) DOMINANCIA DE FEATURES (importancia nativa XGBoost, tipo gain)")
    print("=" * 70)

    resultado_dominancia = verificar_dominancia_de_features()
    print("Top 5 features por importancia:")
    for nombre, importancia in resultado_dominancia["top_5_features"]:
        print(f"  {nombre}: {importancia:.4f}")
    print(f"\nFeature mas importante: {resultado_dominancia['feature_mas_importante']} "
          f"({resultado_dominancia['importancia_relativa_maxima']:.2%} del total)")

    if resultado_dominancia["supera_umbral_dominancia"]:
        print("-> ALERTA: una sola feature supera el 50% de la importancia "
              "total. Revisar si el modelo esta explotando un atajo trivial "
              "del dataset.")
    else:
        print("-> Sin dominancia excesiva de una unica feature: la senal "
              "predictiva esta razonablemente distribuida.")
