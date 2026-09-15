"""
Entrenamiento del Nodo 1 (XGBoost scoring) sobre German Credit.

Estrategia en dos fases, para que la busqueda de hiperparametros y la
fijacion del numero de arboles no compitan por decidir lo mismo:

  Fase 1 (tune_hyperparameters): RandomizedSearchCV, 5-fold, scoring=roc_auc.
  Busca la FORMA del arbol (profundidad, regularizacion, muestreo) con un
  n_estimators fijo y moderado (300), solo para poder comparar configuraciones
  entre si en igualdad de condiciones. n_estimators queda fuera de la rejilla
  a proposito.

  Fase 2 (fit_final_model): con los mejores hiperparametros de la fase 1,
  se fija el numero optimo de arboles mediante xgb.cv (5-fold, con early
  stopping DENTRO de cada fold, promediado entre los 5). Despues se
  reentrena una unica vez sobre el 100% del train con ese numero de
  arboles ya fijo.

  # CORRECTED: la version anterior de esta fase 2 usaba una UNICA
  # particion de validacion (15% del train, ~120 filas) para decidir
  # cuando detener el boosting. Se detecto empiricamente que esto
  # provocaba una parada muy prematura (21 arboles, con
  # learning_rate~0.03) por sobreajuste al ruido de una muestra de
  # validacion pequeña -- una fluctuacion aleatoria del AUC en una
  # ronda temprana bastaba para activar el early stopping, muy por
  # debajo del AUC=0.7952 que la propia fase 1 ya habia demostrado que
  # esta combinacion de hiperparametros alcanza en CV de 5 folds con
  # n_estimators=300 fijo. xgb.cv resuelve esto: el early stopping se
  # evalua dentro de cada uno de los 5 folds y se promedia, dando una
  # estimacion mucho mas estable del numero optimo de arboles, con la
  # misma logica de robustez por CV que ya se aplicaba en la fase 1.
"""
import joblib
import numpy as np
import xgboost as xgb
from scipy.stats import randint, uniform
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

from data_loader import load_german_credit, build_preprocessor, split_data, audit_data_quality

RANDOM_STATE = 42
N_ESTIMATORS_SEARCH = 300   # fijo durante la fase 1, solo para comparar configuraciones
N_ESTIMATORS_MAX = 1000     # limite superior en xgb.cv; el early stopping decide el valor real
EARLY_STOPPING_ROUNDS = 30


def build_pipeline() -> Pipeline:
    preprocessor = build_preprocessor()
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def get_param_distributions() -> dict:
    """
    Espacio de busqueda acotado: con ~800 filas de entrenamiento, arboles
    profundos sobreajustan con facilidad. n_estimators queda FUERA de la
    rejilla a proposito (lo fija xgb.cv en la fase 2). Cada hiperparametro
    conecta directamente con Chen & Guestrin (2016):
      max_depth                    -> profundidad del arbol, acotada por N pequeno
      learning_rate                -> eta (shrinkage)
      subsample / colsample_bytree -> muestreo estocastico (regularizacion implicita)
      min_child_weight             -> minima suma de pesos hessianos por hoja
      gamma                        -> minima ganancia de split (poda)
      reg_alpha / reg_lambda       -> regularizacion L1 / L2
    """
    return {
        "model__max_depth": randint(2, 6),
        "model__learning_rate": uniform(0.01, 0.29),
        "model__subsample": uniform(0.6, 0.4),
        "model__colsample_bytree": uniform(0.6, 0.4),
        "model__min_child_weight": randint(1, 10),
        "model__gamma": uniform(0, 0.5),
        "model__reg_alpha": uniform(0, 1),
        "model__reg_lambda": uniform(0.5, 2),
    }


def tune_hyperparameters(X_train, y_train) -> RandomizedSearchCV:
    """
    Fase 1. n_estimators fijo (300): el objetivo de esta fase es solo
    comparar configuraciones entre si, no fijar el numero final de arboles.
    """
    pipeline = build_pipeline()
    pipeline.set_params(model__n_estimators=N_ESTIMATORS_SEARCH)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=get_param_distributions(),
        n_iter=60,           # 8 hiperparametros, dataset pequeno: 60 configs x 5 folds ya es representativo
        scoring="roc_auc",   # no accuracy: dataset desbalanceado (~70/30)
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    return search


def fit_final_model(best_params_model: dict, X_train, y_train) -> Pipeline:
    """
    Fase 2: fija n_estimators mediante xgb.cv (5-fold, early stopping
    promediado entre folds) y reentrena una unica vez sobre el 100% del
    train con ese numero de arboles ya fijo.

    best_params_model: hiperparametros SIN el prefijo 'model__' (ya
    limpiado antes de llamar a esta funcion).
    """
    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    dtrain = xgb.DMatrix(X_train_t, label=y_train)

    # xgb.cv usa la API nativa de XGBoost (no XGBClassifier): los nombres
    # de algunos parametros difieren ('eta' en vez de 'learning_rate',
    # 'alpha'/'lambda' en vez de 'reg_alpha'/'reg_lambda').
    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": best_params_model["max_depth"],
        "eta": best_params_model["learning_rate"],
        "subsample": best_params_model["subsample"],
        "colsample_bytree": best_params_model["colsample_bytree"],
        "min_child_weight": best_params_model["min_child_weight"],
        "gamma": best_params_model["gamma"],
        "alpha": best_params_model["reg_alpha"],
        "lambda": best_params_model["reg_lambda"],
        "seed": RANDOM_STATE,
    }

    cv_results = xgb.cv(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=N_ESTIMATORS_MAX,
        nfold=5,
        stratified=True,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        metrics="auc",
        seed=RANDOM_STATE,
        verbose_eval=False,
    )

    n_trees_optimo = len(cv_results)
    auc_cv_optimo = cv_results["test-auc-mean"].iloc[-1]
    auc_cv_std = cv_results["test-auc-std"].iloc[-1]
    print(f"xgb.cv (5-fold): {n_trees_optimo} arboles optimos "
          f"(AUC-CV={auc_cv_optimo:.4f} +/- {auc_cv_std:.4f}, "
          f"limite maximo era {N_ESTIMATORS_MAX})")

    # Reentreno final: n_estimators fijo (sin early stopping, ya no hace
    # falta particion de validacion) sobre el 100% del train.
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        n_estimators=n_trees_optimo,
        **best_params_model,
    )
    model.fit(X_train_t, y_train)

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


if __name__ == "__main__":
    df = load_german_credit("data/german-credit-data/german.data")

    print("Auditoria de calidad de datos (German Credit):")
    for k, v in audit_data_quality(df).items():
        print(f"  {k}: {v}")
    print()

    X_train, X_test, y_train, y_test = split_data(df)

    search = tune_hyperparameters(X_train, y_train)
    print(f"Mejor AUC en CV (n_estimators={N_ESTIMATORS_SEARCH} fijo): {search.best_score_:.4f}")
    print(f"Mejores hiperparametros (fase 1): {search.best_params_}")

    best_params_model = {k.replace("model__", "", 1): v for k, v in search.best_params_.items()}

    final_pipeline = fit_final_model(best_params_model, X_train, y_train)

    joblib.dump(final_pipeline, "models/xgb_scoring_pipeline.joblib")
    print("Modelo guardado en models/xgb_scoring_pipeline.joblib")
