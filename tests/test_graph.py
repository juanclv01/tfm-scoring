"""
Test de integracion: valida local accuracy end-to-end (Nodo 1 + Nodo 2).

Ejecutar con: pytest tests/ -v
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from graph import build_graph  # noqa: E402

TOLERANCE = 1e-4

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
    """sum(shap_values) + base_value debe reconstruir proba_default."""
    app = build_graph()
    resultado = app.invoke({"client_data": CLIENTE_EJEMPLO})

    suma_shap = np.sum(resultado["shap_values"])
    reconstruccion = resultado["base_value"] + suma_shap

    assert abs(reconstruccion - resultado["proba_default"]) < TOLERANCE, (
        f"Local accuracy violada: reconstruccion={reconstruccion:.6f} "
        f"vs proba_default={resultado['proba_default']:.6f}"
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
