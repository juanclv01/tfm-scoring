"""
Calcula y persiste dos umbrales de decision (Nodo 1), ambos derivados
empiricamente del test set del modelo ya entrenado -- ninguno se hardcodea:

  1. Umbral de APROBACION/RECHAZO: el que minimiza el coste esperado segun
     la matriz 5:1 oficial de UCI para German Credit (evaluate.py). El
     umbral 0.5 no tiene significado de negocio para este problema -- ver
     docstring de evaluate.py ("evaluadas en el UMBRAL OPTIMO DE COSTE,
     no en 0.5").

  2. Umbral MODERADO/ALTO (banda de riesgo, solo para narrativas): dentro
     del subconjunto ya rechazado (proba_default >= umbral de aprobacion),
     se toma la MEDIANA de proba_default. Esto divide a los clientes
     rechazados en dos mitades iguales -- "moderado" (mitad con menos
     riesgo dentro de los rechazados) y "alto" (mitad con mas riesgo) --
     en vez de fijar un corte redondo arbitrario (p.ej. 65%) sin respaldo
     empirico. Igual que el umbral de aprobacion, se recalcula con el
     modelo actual: no requiere ninguna cifra fija en el codigo.

Ambos se calculan SOBRE EL TEST SET (nunca sobre el train, para no
filtrar informacion de la propia decision de negocio hacia el conjunto
que ya se uso para fijar hiperparametros/n_estimators), en la misma
pasada de predict_proba() para no duplicar el calculo.

Este script no aparecia en la guia de implementacion original: node_scoring()
necesitaba estos umbrales para poder anadir 'aprobado' y 'nivel_riesgo' al
estado del Nodo 1 (imprescindibles para las narrativas de ejemplo de
Nodo 3, que necesitan saber tanto si la solicitud fue aprobada como su
banda cualitativa de riesgo), y nada lo generaba. Mismo patron que
prepare_background.py: script offline independiente, porque calcular
estos umbrales requiere recargar el dataset completo y rehacer el split
-- coste que no tiene sentido pagar en cada arranque del grafo en
produccion.

IMPORTANTE: ninguno de estos dos numeros debe aparecer nunca en el prompt
del NARRATOR ni en la narrativa final -- node_scoring() los usa para
calcular 'aprobado' y 'nivel_riesgo' (etiquetas ya resueltas), y es
UNICAMENTE esa etiqueta la que llega al LLM. La clasificacion numerica es
responsabilidad del codigo, nunca del LLM (ver scoring.clasificar_riesgo).

Ejecutar SIEMPRE despues de train_model.py y ANTES de graph.py.
"""
import joblib
import numpy as np

from data_loader import load_german_credit, split_data
from evaluate import find_cost_optimal_threshold

if __name__ == "__main__":
    model = joblib.load("models/xgb_scoring_pipeline.joblib")

    df = load_german_credit("data/german-credit-data/german.data")
    _, X_test, _, y_test = split_data(df)

    y_proba = model.predict_proba(X_test)[:, 1]
    resultado = find_cost_optimal_threshold(y_test, y_proba)
    umbral_aprobacion = resultado["umbral_optimo"]

    joblib.dump(umbral_aprobacion, "models/decision_threshold.joblib")

    # Umbral moderado/alto: mediana de proba_default DENTRO de los
    # rechazados (proba_default >= umbral_aprobacion), no sobre todo el
    # test set -- lo que se quiere dividir en dos mitades es la poblacion
    # ya rechazada, no la poblacion general (que incluiria aprobados).
    rechazados_mask = y_proba >= umbral_aprobacion
    n_rechazados = int(rechazados_mask.sum())
    if n_rechazados < 2:
        raise ValueError(
            f"Solo hay {n_rechazados} instancia(s) rechazada(s) en el test "
            f"set -- insuficiente para calcular una mediana representativa "
            f"del umbral moderado/alto. Revisa el modelo o el umbral de "
            f"aprobacion antes de continuar."
        )
    umbral_moderado_alto = float(np.median(y_proba[rechazados_mask]))

    joblib.dump(umbral_moderado_alto, "models/risk_band_threshold.joblib")

    print(f"Umbral optimo de aprobacion (proba_default): {umbral_aprobacion:.4f}")
    print(f"  -> equivale a score de aprobacion: {1000 * (1 - umbral_aprobacion):.1f} / 1000")
    print(f"Coste esperado en este umbral: {resultado['coste_esperado_optimo']:.4f}")
    print(f"Coste esperado en umbral 0.5:  {resultado['coste_esperado_umbral_0.5']:.4f}")
    print()
    print(f"Umbral moderado/alto (mediana de {n_rechazados} rechazados): "
          f"{umbral_moderado_alto:.4f}")
    print(f"  -> equivale a score: {1000 * (1 - umbral_moderado_alto):.1f} / 1000")
    print()
    print("Bandas de riesgo resultantes (proba_default):")
    print(f"  bajo:      [0.0000, {umbral_aprobacion:.4f})")
    print(f"  moderado:  [{umbral_aprobacion:.4f}, {umbral_moderado_alto:.4f})")
    print(f"  alto:      [{umbral_moderado_alto:.4f}, 1.0000]")
    print()
    print("Umbrales guardados en models/decision_threshold.joblib y "
          "models/risk_band_threshold.joblib")
