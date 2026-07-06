"""
Comparacion de metricas (AUC-ROC, Gini, KS, Brier, coste esperado) entre
los tres modelos entrenados sobre datasets distintos. Ejecutar DESPUES de:
  python src/train_model.py             (German Credit)
  python src/train_model_credit_card.py (Credit Card)
  python src/train_model_home_credit.py (Home Credit)

Esta tabla es la evidencia cuantitativa de que evaluate.py (AUC/Gini/KS)
se comporta de forma consistente independientemente del dataset -- la
"validacion cruzada de metricas" de la tabla de datasets del TFM.
"""
import joblib

from data_loader import load_german_credit, split_data as split_german
from data_loader_credit_card import load_credit_card, split_data as split_credit_card
from data_loader_home_credit import load_home_credit, split_data as split_home_credit
from evaluate import evaluate_model


# # CORRECTED: get_german_credit_test_set(), get_credit_card_test_set() y
# get_home_credit_test_set() eran 3 funciones casi identicas (cargar,
# dividir, devolver X_test/y_test), difiriendo solo en que par
# load/split reciben. 3 ocurrencias de la misma logica es el umbral que
# justifica una funcion compartida (ver restriccion de no anadir
# abstracciones salvo que eliminen duplicacion real de 3+ ocurrencias).
def _get_test_set(load_fn, split_fn):
    df = load_fn()
    _, X_test, _, y_test = split_fn(df)
    return X_test, y_test


DATASETS = [
    {
        "nombre": "German Credit",
        "modelo": "models/xgb_scoring_pipeline.joblib",
        "cargar_test": lambda: _get_test_set(load_german_credit, split_german),
        # Matriz de coste 5:1 oficial de UCI para este dataset especifico.
        "cost_matrix_is_official": True,
        "cost_aprobar_malo": 5,
        "cost_rechazar_bueno": 1,
    },
    {
        "nombre": "Default Credit Card",
        "modelo": "models/xgb_credit_card_pipeline.joblib",
        "cargar_test": lambda: _get_test_set(load_credit_card, split_credit_card),
        # Sin matriz de coste oficial documentada por UCI/Kaggle para este
        # dataset. Se reutiliza el mismo ratio 5:1 como supuesto
        # simplificador, NO como hecho verificado -- marcado explicitamente
        # via cost_matrix_is_official=False para que la tabla de resultados
        # no de a entender lo contrario.
        "cost_matrix_is_official": False,
        "cost_aprobar_malo": 5,
        "cost_rechazar_bueno": 1,
    },
    {
        "nombre": "Home Credit Default",
        "modelo": "models/xgb_home_credit_pipeline.joblib",
        "cargar_test": lambda: _get_test_set(load_home_credit, split_home_credit),
        "cost_matrix_is_official": False,
        "cost_aprobar_malo": 5,
        "cost_rechazar_bueno": 1,
    },
]


def main():
    print(f"{'Dataset':<22}{'AUC-ROC':>9}{'Gini':>9}{'KS':>9}{'Brier':>9}{'Coste ofic.':>13}")
    print("-" * 71)

    for entry in DATASETS:
        try:
            model = joblib.load(entry["modelo"])
        except FileNotFoundError:
            print(f"{entry['nombre']:<22}{'(modelo no entrenado aun)':>41}")
            continue

        X_test, y_test = entry["cargar_test"]()
        metrics = evaluate_model(
            model, X_test, y_test,
            cost_aprobar_malo=entry["cost_aprobar_malo"],
            cost_rechazar_bueno=entry["cost_rechazar_bueno"],
            cost_matrix_is_official=entry["cost_matrix_is_official"],
        )

        print(
            f"{entry['nombre']:<22}"
            f"{metrics['auc_roc']:>9.4f}"
            f"{metrics['gini']:>9.4f}"
            f"{metrics['ks_statistic']:>9.4f}"
            f"{metrics['brier_score']:>9.4f}"
            f"{'si' if metrics['cost_matrix_is_official'] else 'NO (supuesto)':>13}"
        )


if __name__ == "__main__":
    main()
