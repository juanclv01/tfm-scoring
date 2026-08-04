"""
Test de integracion: valida local accuracy end-to-end (Nodo 1 + Nodo 2).

Ejecutar con: pytest tests/ -v
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from graph import build_graph  # noqa: E402

TOLERANCE_SCORE = 0.1     # CORRECTED: shap_values/base_value ahora estan en
                           # puntos de score (nodes.py convierte de espacio
                           # de proba_default a score en Nodo 2). La tolerancia
                           # que antes era 1e-4 en espacio de proba_default
                           # equivale a 1e-4 * 1000 = 0.1 puntos de score.

CLIENTE_EJEMPLO = {
    "checking_status": "A11", "duration": 24, "credit_history": "A32",
    "purpose": "A43", "credit_amount": 3500, "savings_status": "A61",
    "employment": "A73", "installment_rate": 3, "personal_status": "A93",
    "other_parties": "A101", "residence_since": 2,
    "property_magnitude": "A121", "age": 34,
    "other_payment_plans": "A143", "housing": "A152",
    "existing_credits": 1, "job": "A173", "num_dependents": 1,
    "own_telephone": "A192", "foreign_worker": "A201",
}


def test_local_accuracy_holds():
    """
    sum(shap_values) + base_value debe reconstruir el score continuo.

    # CORRECTED: nodes.py ahora convierte shap_values y base_value de
    # espacio de proba_default a PUNTOS DE SCORE antes de devolverlos en
    # el estado (shap_value > 0 -> sube el score, convencion mas
    # intuitiva que la que devuelve TreeExplainer por defecto). Local
    # accuracy sigue cumpliendose matematicamente -- es una transformacion
    # lineal, no rompe la propiedad -- pero ya no se verifica contra
    # proba_default directamente, sino contra el score CONTINUO derivado
    # de el (1000 * (1 - proba_default)). No se compara contra
    # resultado["score"] porque ese ya lleva el redondeo a entero de
    # scoring.probability_to_score(), lo que introduciria hasta 0.5 puntos
    # de discrepancia ajena a la propiedad matematica que este test
    # realmente quiere comprobar.
    """
    app = build_graph()
    resultado = app.invoke({"client_data": CLIENTE_EJEMPLO})

    suma_shap = np.sum(resultado["shap_values"])
    reconstruccion = resultado["base_value"] + suma_shap

    score_continuo = 1000 * (1 - resultado["proba_default"])

    assert abs(reconstruccion - score_continuo) < TOLERANCE_SCORE, (
        f"Local accuracy violada (espacio de score): "
        f"reconstruccion={reconstruccion:.4f} vs "
        f"score_continuo={score_continuo:.4f}"
    )


def test_score_en_rango_valido():
    """El score siempre debe estar entre 0 y 1000."""
    app = build_graph()
    resultado = app.invoke({"client_data": CLIENTE_EJEMPLO})
    assert 0 <= resultado["score"] <= 1000


def test_top_features_no_vacio():
    """El nodo de explicabilidad debe devolver siempre 5 factores."""
    app = build_graph()
    resultado = app.invoke({"client_data": CLIENTE_EJEMPLO})
    assert len(resultado["top_features"]) == 5


def test_aprobado_es_booleano_y_consistente_con_umbral():
    """
    'aprobado' debe ser un bool puro (no np.bool_) y su valor debe ser
    consistente con proba_default y el umbral de coste persistido en
    models/decision_threshold.joblib -- NO con el corte ingenuo 0.5.

    # CORRECTED: no se compara contra proba_default < 0.5 porque ese
    # umbral no tiene significado de negocio para este problema (ver
    # docstring de evaluate.py); se carga el mismo umbral que usa
    # node_scoring() para no duplicar el criterio de decision en el test.
    """
    import joblib

    app = build_graph()
    resultado = app.invoke({"client_data": CLIENTE_EJEMPLO})

    assert isinstance(resultado["aprobado"], bool)

    umbral_optimo = joblib.load("models/decision_threshold.joblib")
    esperado = resultado["proba_default"] < umbral_optimo
    assert resultado["aprobado"] == esperado, (
        f"aprobado={resultado['aprobado']} no coincide con el umbral de "
        f"coste optimo ({umbral_optimo:.4f}) para "
        f"proba_default={resultado['proba_default']:.4f}"
    )
