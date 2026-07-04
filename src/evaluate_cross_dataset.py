"""
Comparacion de metricas (AUC-ROC, Gini, KS) entre los tres modelos
entrenados sobre datasets distintos. Ejecutar DESPUES de:
  python src/train_model.py             (German Credit)
  python src/train_model_credit_card.py (Credit Card)
  python src/train_model_home_credit.py (Home Credit)

Esta tabla es la evidencia cuantitativa de que evaluate.py (AUC/Gini/KS)
se comporta de forma consistente independientemente del dataset -- la
"validacion cruzada de metricas" de la tabla de datasets del TFM.
"""
import joblib

from data_loader import load_german_credit, split_data as split_german
from data_loader_credit_card import load_credit_card
from data_loader_home_credit import load_home_credit
from dataset_utils import split_data
from evaluate import evaluate_model


def get_german_credit_test_set():
    df = load_german_credit()
    _, X_test, _, y_test = split_german(df)
    return X_test, y_test


def get_credit_card_test_set():
    df = load_credit_card()
    _, X_test, _, y_test = split_data(df, target_col="target")
    return X_test, y_test


def get_home_credit_test_set():
    df = load_home_credit()
    _, X_test, _, y_test = split_data(df, target_col="target")
    return X_test, y_test


DATASETS = [
    {
        "nombre": "German Credit",
        "modelo": "models/xgb_scoring_pipeline.joblib",
        "cargar_test": get_german_credit_test_set,
    },
    {
        "nombre": "Default Credit Card",
        "modelo": "models/xgb_credit_card_pipeline.joblib",
        "cargar_test": get_credit_card_test_set,
    },
    {
        "nombre": "Home Credit Default",
        "modelo": "models/xgb_home_credit_pipeline.joblib",
        "cargar_test": get_home_credit_test_set,
    },
]


def main():
    print(f"{'Dataset':<22}{'AUC-ROC':>10}{'Gini':>10}{'KS':>10}")
    print("-" * 52)

    for entry in DATASETS:
        try:
            model = joblib.load(entry["modelo"])
        except FileNotFoundError:
            print(f"{entry['nombre']:<22}{'(modelo no entrenado aun)':>32}")
            continue

        X_test, y_test = entry["cargar_test"]()
        metrics = evaluate_model(model, X_test, y_test)

        print(
            f"{entry['nombre']:<22}"
            f"{metrics['auc_roc']:>10.4f}"
            f"{metrics['gini']:>10.4f}"
            f"{metrics['ks_statistic']:>10.4f}"
        )


if __name__ == "__main__":
    main()
