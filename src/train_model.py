"""
Entrenamiento del Nodo 1 (XGBoost scoring) sobre German Credit.

Estrategia en dos fases, para que la busqueda de hiperparametros y el
early stopping no compitan por decidir lo mismo (el numero de arboles):

  Fase 1 (tune_hyperparameters): RandomizedSearchCV, 5-fold, scoring=roc_auc.
  Busca la FORMA del arbol (profundidad, regularizacion, muestreo) con un
  n_estimators fijo y moderado (300), solo para poder comparar configuraciones
  entre si en igualdad de condiciones. n_estimators queda fuera de la rejilla
  a proposito.

  Fase 2 (fit_final_model): con los mejores hiperparametros de la fase 1,
  refit con early stopping sobre una particion de validacion tomada del
  propio TRAIN (nunca del test) para fijar el n_estimators optimo. Despues
  se reentrena una ultima vez con ese n_estimators fijo sobre el 100% del
  train, para no desperdiciar ese porcentaje reservado en el modelo final
  que se guarda en disco.

  # CORRECTED: la version anterior buscaba n_estimators dentro de
  # RandomizedSearchCV (100-500) sin early stopping -- dos mecanismos que
  # persiguen el mismo objetivo (evitar que el boosting siga anadiendo
  # arboles quando ya no aporta) sin comunicarse entre si. Con ~800 filas
  # de train el early stopping es mas barato y mas preciso que dejar que
  # la busqueda aleatoria "adivine" el numero de arboles.
"""
import joblib
from scipy.stats import randint, uniform
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from data_loader import load_german_credit, build_preprocessor, split_data, audit_data_quality

RANDOM_STATE = 42
N_ESTIMATORS_SEARCH = 300   # fijo durante la fase 1, solo para comparar configuraciones
N_ESTIMATORS_MAX = 1000     # limite superior en la fase 2; early stopping decide el valor real
EARLY_STOPPING_ROUNDS = 30
VALIDATION_SIZE = 0.15      # particion interna del TRAIN para early stopping (no toca el test)


def build_pipeline(early_stopping_rounds=None) -> Pipeline:
    preprocessor = build_preprocessor()
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        early_stopping_rounds=early_stopping_rounds,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def get_param_distributions() -> dict:
    """
    Espacio de busqueda acotado: con ~800 filas de entrenamiento, arboles
    profundos sobreajustan con facilidad. n_estimators queda FUERA de la
    rejilla a proposito (lo fija el early stopping de la fase 2). Cada
    hiperparametro conecta directamente con Chen & Guestrin (2016):
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
    Fase 1. Sin early stopping: el early stopping necesita un eval_set fijo
    por fold, lo que complicaria la CV sin aportar nada aqui (el objetivo de
    esta fase es solo comparar configuraciones entre si, no fijar n_estimators).
    """
    pipeline = build_pipeline(early_stopping_rounds=None)
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
    Fase 2: fija n_estimators via early stopping y reentrena sobre el 100%
    del train con ese numero fijo de arboles.

    best_params_model: hiperparametros de XGBClassifier SIN el prefijo
    'model__' (ya limpiado antes de llamar a esta funcion).
    """
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=VALIDATION_SIZE,
        stratify=y_train, random_state=RANDOM_STATE,
    )

    # El preprocesador se ajusta aqui solo sobre X_fit (85% del train) para
    # que el eval_set de early stopping (X_val) no participe en absoluto en
    # el ajuste del OneHotEncoder -- evita cualquier fuga, por pequena que sea.
    preprocessor = build_preprocessor()
    X_fit_t = preprocessor.fit_transform(X_fit)
    X_val_t = preprocessor.transform(X_val)

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        n_estimators=N_ESTIMATORS_MAX,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        **best_params_model,
    )
    model.fit(X_fit_t, y_fit, eval_set=[(X_val_t, y_val)], verbose=False)

    n_trees_optimo = model.best_iteration + 1
    print(f"Early stopping: {n_trees_optimo} arboles "
          f"(limite maximo era {N_ESTIMATORS_MAX})")

    # Reentrenar con n_estimators fijo sobre el 100% del train: el 15%
    # reservado arriba solo servia para decidir CUANTOS arboles usar, no
    # tiene sentido dejarlo fuera del modelo que finalmente se guarda.
    preprocessor_full = build_preprocessor()
    X_train_t = preprocessor_full.fit_transform(X_train)
    model.set_params(n_estimators=n_trees_optimo, early_stopping_rounds=None)
    model.fit(X_train_t, y_train, verbose=False)

    return Pipeline(steps=[("preprocessor", preprocessor_full), ("model", model)])


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

    # search.best_params_ trae claves con prefijo 'model__' (nombre del step
    # en el Pipeline de la fase 1); XGBClassifier standalone de la fase 2
    # necesita las claves sin ese prefijo.
    best_params_model = {k.replace("model__", "", 1): v for k, v in search.best_params_.items()}

    final_pipeline = fit_final_model(best_params_model, X_train, y_train)

    joblib.dump(final_pipeline, "models/xgb_scoring_pipeline.joblib")
    print("Modelo guardado en models/xgb_scoring_pipeline.joblib")
