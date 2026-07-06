"""
Metricas de evaluacion del Nodo 1: AUC-ROC, Gini, KS statistic,
coste esperado (matriz de coste) y Brier score (calibracion).
"""
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss

# Matriz de coste oficial del German Credit Dataset (UCI): aprobar a un
# cliente que en realidad impagara cuesta 5x mas que rechazar a uno que
# si habria pagado. target: 0=bueno, 1=malo.
#
# # CORRECTED: estas constantes se usaban directamente dentro de
# expected_cost()/find_cost_optimal_threshold(), aplicandose de forma
# implicita e identica a los tres datasets (German Credit, Credit Card,
# Home Credit) via evaluate_model(). Esa 5:1 SOLO esta documentada
# oficialmente para German Credit -- aplicarla sin mas a Credit Card o
# Home Credit era una generalizacion no justificada (bug de logica, no
# solo de estilo). Se mantienen como valor por defecto (German Credit
# sigue funcionando exactamente igual sin cambiar ninguna llamada
# existente), pero ahora son parametros que evaluate_cross_dataset.py
# puede/debe sobrescribir para Credit Card y Home Credit, documentando
# explicitamente que se usa el mismo ratio como supuesto simplificador
# a falta de una matriz de coste oficial propia de esos dos datasets.
COST_APROBAR_MALO_DEFAULT = 5    # falso negativo: predices bueno, era malo
COST_RECHAZAR_BUENO_DEFAULT = 1  # falso positivo: predices malo, era bueno


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


def calibration_score(y_true, y_proba) -> float:
    """
    Brier score: error cuadratico medio entre probabilidad predicha y
    resultado real (0 o 1). A diferencia de AUC/Gini/KS (que solo miden
    si el ORDEN de riesgo es correcto), el Brier score mide si el VALOR
    de la probabilidad es fiable en si mismo. Importa especificamente
    para este pipeline porque el Nodo 2 (SHAP) asume que proba_default
    es una probabilidad bien calibrada -- si no lo es, la propiedad de
    local accuracy sigue cumpliendose matematicamente, pero el numero
    que se explica ya no refleja el riesgo real del cliente.
    Rango: 0 (calibracion perfecta) a 1 (peor caso).
    """
    return float(brier_score_loss(y_true, y_proba))


def expected_cost(
    y_true, y_pred,
    cost_aprobar_malo: float = COST_APROBAR_MALO_DEFAULT,
    cost_rechazar_bueno: float = COST_RECHAZAR_BUENO_DEFAULT,
) -> float:
    """
    Coste esperado por cliente segun una matriz de coste (por defecto,
    la oficial de UCI para German Credit -- ver nota sobre parametrizacion
    arriba). NO debe usarse para entrenar el modelo (distorsionaria las
    probabilidades que SHAP necesita calibradas); se usa solo para
    evaluar la calidad de la decision final aprobar/rechazar en un
    umbral dado.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    falsos_negativos = np.sum((y_true == 1) & (y_pred == 0))  # aprobado un malo
    falsos_positivos = np.sum((y_true == 0) & (y_pred == 1))  # rechazado un bueno

    coste_total = (
        falsos_negativos * cost_aprobar_malo
        + falsos_positivos * cost_rechazar_bueno
    )
    return float(coste_total / len(y_true))


def find_cost_optimal_threshold(
    y_true, y_proba, n_steps: int = 200,
    cost_aprobar_malo: float = COST_APROBAR_MALO_DEFAULT,
    cost_rechazar_bueno: float = COST_RECHAZAR_BUENO_DEFAULT,
) -> dict:
    """
    Barre umbrales de 0 a 1 y devuelve el que minimiza el coste esperado,
    en lugar de asumir por defecto el umbral de 0.5. Esto es lo que
    traduce la matriz de coste en una decision de negocio real, sin
    tocar el entrenamiento del modelo ni sus probabilidades.
    """
    mejor_umbral, mejor_coste = 0.5, float("inf")
    for umbral in np.linspace(0.01, 0.99, n_steps):
        y_pred = (y_proba >= umbral).astype(int)
        coste = expected_cost(y_true, y_pred, cost_aprobar_malo, cost_rechazar_bueno)
        if coste < mejor_coste:
            mejor_umbral, mejor_coste = float(umbral), coste

    coste_umbral_05 = expected_cost(
        y_true, (y_proba >= 0.5).astype(int), cost_aprobar_malo, cost_rechazar_bueno
    )

    return {
        "umbral_optimo": mejor_umbral,
        "coste_esperado_optimo": mejor_coste,
        "coste_esperado_umbral_0.5": coste_umbral_05,
    }


def evaluate_model(
    model, X_test, y_test,
    cost_aprobar_malo: float = COST_APROBAR_MALO_DEFAULT,
    cost_rechazar_bueno: float = COST_RECHAZAR_BUENO_DEFAULT,
    cost_matrix_is_official: bool = True,
) -> dict:
    """
    # CORRECTED: acepta ahora cost_aprobar_malo/cost_rechazar_bueno como
    parametros opcionales (antes eran constantes fijas del modulo,
    aplicadas identicamente a cualquier dataset que llamara a esta
    funcion). cost_matrix_is_official documenta si el ratio usado tiene
    respaldo documental oficial para el dataset evaluado (True solo para
    German Credit) o si es un supuesto simplificador (False para Credit
    Card / Home Credit, que no tienen matriz de coste publicada por UCI/Kaggle).
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    resultado_coste = find_cost_optimal_threshold(
        y_test, y_proba, cost_aprobar_malo=cost_aprobar_malo,
        cost_rechazar_bueno=cost_rechazar_bueno,
    )
    return {
        "auc_roc": roc_auc_score(y_test, y_proba),
        "gini": gini_coefficient(y_test, y_proba),
        "ks_statistic": ks_statistic(y_test, y_proba),
        "brier_score": calibration_score(y_test, y_proba),
        "cost_matrix_is_official": cost_matrix_is_official,
        **resultado_coste,
    }


if __name__ == "__main__":
    import joblib
    from data_loader import load_german_credit, split_data

    model = joblib.load("models/xgb_scoring_pipeline.joblib")
    df = load_german_credit("data/german-credit-data/german.data")
    _, X_test, _, y_test = split_data(df)

    # German Credit: cost_matrix_is_official=True por defecto (matriz UCI real).
    metrics = evaluate_model(model, X_test, y_test)
    for k, v in metrics.items():
        print(f"{k}: {v}")
