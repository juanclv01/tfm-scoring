"""
Entrenamiento del modelo sobre "Home Credit Default Risk".

Diferencia deliberada frente a train_model.py y train_model_credit_card.py:
con 307.511 filas, una busqueda de 100 combinaciones x 5 folds (500
entrenamientos completos) puede tardar horas. Aqui se reduce n_iter y
el numero de folds, priorizando obtener una metrica de referencia en
tiempo razonable sobre encontrar el optimo absoluto -- coherente con el
objetivo de esta fase ("prueba de escala", no maximizar rendimiento).

Se mide el tiempo de ejecucion explicitamente: ese numero es en si mismo
un resultado a reportar en la memoria (?el pipeline es viable a escala
real, o el tiempo de entrenamiento lo hace impractico para produccion?).
"""
import time

import joblib
from scipy.stats import randint, uniform
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

from data_loader_home_credit import (
    load_home_credit,
    infer_feature_types,
    build_preprocessor_home_credit,
)
from dataset_utils import split_data

RANDOM_STATE = 42


def build_pipeline(categorical_features: list, numeric_features: list) -> Pipeline:
    preprocessor = build_preprocessor_home_credit(categorical_features, numeric_features)
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",  # algoritmo aproximado (Sec. 3 de Chen & Guestrin):
                             # imprescindible a este volumen, el exact greedy
                             # de German Credit no escala a 300k filas
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def get_param_distributions() -> dict:
    """Espacio reducido respecto a German Credit: menos combinaciones
    a explorar dado el coste de cada entrenamiento a esta escala."""
    return {
        "model__n_estimators": randint(100, 300),
        "model__max_depth": randint(3, 8),
        "model__learning_rate": uniform(0.01, 0.29),
        "model__subsample": uniform(0.6, 0.4),
        "model__colsample_bytree": uniform(0.6, 0.4),
    }


def tune_model(X_train, y_train, categorical_features, numeric_features) -> RandomizedSearchCV:
    pipeline = build_pipeline(categorical_features, numeric_features)
    # 3 folds en lugar de 5, y n_iter=15 en lugar de 100: a este volumen,
    # cada entrenamiento individual ya es representativo por si solo.
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=get_param_distributions(),
        n_iter=15,
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=2,
    )
    search.fit(X_train, y_train)
    return search


if __name__ == "__main__":
    t0 = time.time()

    df = load_home_credit()
    categorical, numeric = infer_feature_types(df)
    X_train, X_test, y_train, y_test = split_data(df, target_col="target")

    t_load = time.time()
    print(f"Carga y split: {t_load - t0:.1f} s ({len(df)} filas)")

    search = tune_model(X_train, y_train, categorical, numeric)

    t_train = time.time()
    print(f"Busqueda de hiperparametros: {t_train - t_load:.1f} s")
    print(f"Mejor AUC en CV: {search.best_score_:.4f}")
    print(f"Mejores hiperparametros: {search.best_params_}")
    print(f"Tiempo total: {t_train - t0:.1f} s")

    joblib.dump(search.best_estimator_, "models/xgb_home_credit_pipeline.joblib")
    print("Modelo guardado en models/xgb_home_credit_pipeline.joblib")
