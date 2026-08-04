"""
Calcula y persiste el umbral de decision de aprobacion/rechazo (Nodo 1).

Justificacion: el umbral 0.5 no tiene significado de negocio para este
problema -- ver docstring de evaluate.py ("evaluadas en el UMBRAL OPTIMO DE
COSTE, no en 0.5"). En su lugar se usa el umbral que minimiza el coste
esperado segun la matriz 5:1 oficial de UCI para German Credit, calculado
por evaluate.find_cost_optimal_threshold() SOBRE EL TEST SET (nunca sobre
el train, para no filtrar informacion de la propia decision de negocio
hacia el conjunto que ya se uso para fijar hiperparametros/n_estimators).

Este script no aparecia en la guia de implementacion original: node_scoring()
necesitaba un umbral de aprobacion para poder anadir 'aprobado' al estado
del Nodo 1 (imprescindible para poder construir las narrativas de ejemplo
de Nodo 3, que necesitan saber si la solicitud fue aprobada o rechazada), y
nada lo generaba. Mismo patron que prepare_background.py: script offline
independiente, porque calcular el umbral optimo requiere recargar el
dataset completo y rehacer el split -- coste que no tiene sentido pagar en
cada arranque del grafo en produccion.

Ejecutar SIEMPRE despues de train_model.py y ANTES de graph.py.
"""
import joblib

from data_loader import load_german_credit, split_data
from evaluate import find_cost_optimal_threshold

if __name__ == "__main__":
    model = joblib.load("models/xgb_scoring_pipeline.joblib")

    df = load_german_credit("data/german-credit-data/german.data")
    _, X_test, _, y_test = split_data(df)

    y_proba = model.predict_proba(X_test)[:, 1]
    resultado = find_cost_optimal_threshold(y_test, y_proba)
    umbral_optimo = resultado["umbral_optimo"]

    joblib.dump(umbral_optimo, "models/decision_threshold.joblib")

    print(f"Umbral optimo de decision (proba_default): {umbral_optimo:.4f}")
    print(f"  -> equivale a score de aprobacion: {1000 * (1 - umbral_optimo):.1f} / 1000")
    print(f"Coste esperado en este umbral: {resultado['coste_esperado_optimo']:.4f}")
    print(f"Coste esperado en umbral 0.5:  {resultado['coste_esperado_umbral_0.5']:.4f}")
    print("Umbral guardado en models/decision_threshold.joblib")
