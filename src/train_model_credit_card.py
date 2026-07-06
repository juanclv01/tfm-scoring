"""
Entrenamiento del modelo sobre "Default of Credit Card Clients".
Mismo patron exacto que train_model.py (German Credit); solo cambia
el loader y el preprocesador. Ver train_model.py para la explicacion
detallada de cada hiperparametro -- no se repite aqui.
"""
import joblib
from scipy.stats import randint, uniform
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

from data_loader_credit_card import (
    load_credit_card,
    build_preprocessor_credit_card,
    split_data,
    audit_data_quality,
)

RANDOM_STATE = 42


def build_pipeline() -> Pipeline:
    preprocessor = build_preprocessor_credit_card()
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def get_param_distributions() -> dict:
    """Mismo espacio de busqueda que German Credit: 30.000 filas es
    suficiente margen para no reducirlo, pero tampoco hace falta ampliarlo."""
    return {
        "model__n_estimators": randint(100, 500),
        "model__max_depth": randint(2, 8),
        "model__learning_rate": uniform(0.01, 0.29),
        "model__subsample": uniform(0.6, 0.4),
        "model__colsample_bytree": uniform(0.6, 0.4),
        "model__min_child_weight": randint(1, 10),
        "model__gamma": uniform(0, 0.5),
        "model__reg_alpha": uniform(0, 1),
        "model__reg_lambda": uniform(0.5, 2),
    }


def tune_model(X_train, y_train) -> RandomizedSearchCV:
    pipeline = build_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=get_param_distributions(),
        n_iter=100,
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    return search


if __name__ == "__main__":
    df = load_credit_card()

    # CORRECTED: mismo gap de integracion que en train_model.py -- la
    # auditoria existia pero no formaba parte del flujo de entrenamiento.
    print("Auditoria de calidad de datos (Credit Card):")
    for k, v in audit_data_quality(df).items():
        print(f"  {k}: {v}")
    print()

    X_train, X_test, y_train, y_test = split_data(df)

    search = tune_model(X_train, y_train)
    print(f"Mejor AUC en CV: {search.best_score_:.4f}")
    print(f"Mejores hiperparametros: {search.best_params_}")

    joblib.dump(search.best_estimator_, "models/xgb_credit_card_pipeline.joblib")
    print("Modelo guardado en models/xgb_credit_card_pipeline.joblib")
