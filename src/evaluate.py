"""
Metricas de evaluacion del Nodo 1: AUC-ROC, Gini, KS statistic.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def gini_coefficient(y_true, y_proba) -> float:
    """Gini = 2 * AUC - 1 (equivalente al Accuracy Ratio de Basilea II)."""
    auc = roc_auc_score(y_true, y_proba)
    return 2 * auc - 1


def ks_statistic(y_true, y_proba) -> float:
    """
    Kolmogorov-Smirnov: maxima distancia vertical entre las curvas de
    distribucion acumulada de 'buenos' y 'malos' pagadores segun el score.
    Es la metrica que reporta QuickBooks Capital en el caso de produccion
    citado en la bibliografia (industria financiera).
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    return float(np.max(np.abs(tpr - fpr)))


def evaluate_model(model, X_test, y_test) -> dict:
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "auc_roc": roc_auc_score(y_test, y_proba),
        "gini": gini_coefficient(y_test, y_proba),
        "ks_statistic": ks_statistic(y_test, y_proba),
    }


if __name__ == "__main__":
    import joblib
    from data_loader import load_german_credit, split_data

    model = joblib.load("models/xgb_scoring_pipeline.joblib")
    df = load_german_credit("data/german-credit-data/german.data")
    _, X_test, _, y_test = split_data(df)

    metrics = evaluate_model(model, X_test, y_test)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
