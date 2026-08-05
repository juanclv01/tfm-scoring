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


def test_top_features_sin_variables_categoricas_duplicadas():
    """
    Ninguna variable original (feature_raw) debe aparecer dos veces en
    top_features -- de lo contrario, para una variable categorica, dos
    entradas del mismo feature_raw significarian dos categorias
    mutuamente excluyentes narradas como si el cliente estuviera en
    ambas simultaneamente a la vez (bug real, detectado por revision
    manual de narrativas de ejemplo: 'checking_status' aparecia como
    'sin cuenta corriente' Y 'saldo negativo en cuenta corriente' para
    el mismo cliente). Corregido agregando por variable original en
    node_explainability() antes de seleccionar el top-5 -- este test
    fija esa propiedad para que no se rompa sin que ningun test lo note.
    """
    app = build_graph()
    resultado = app.invoke({"client_data": CLIENTE_EJEMPLO})

    variables_originales = [f["feature_raw"] for f in resultado["top_features"]]
    assert len(variables_originales) == len(set(variables_originales)), (
        f"Variables duplicadas en top_features: {variables_originales}"
    )

    # Ademas, feature_value debe coincidir siempre con el valor REAL del
    # cliente para features categoricas (nunca el codigo de una dummy
    # que pudiera valer 0 para este cliente).
    from data_loader import decode_feature_value, CATEGORICAL_FEATURES

    for f in resultado["top_features"]:
        if f["feature_raw"] in CATEGORICAL_FEATURES:
            valor_esperado = decode_feature_value(
                f["feature_raw"], CLIENTE_EJEMPLO[f["feature_raw"]]
            )
            assert f["feature_value"] == valor_esperado, (
                f"feature_value={f['feature_value']!r} no coincide con el "
                f"valor real del cliente para {f['feature_raw']!r} "
                f"(esperado: {valor_esperado!r})"
            )


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


def test_nivel_riesgo_valido_y_consistente_con_umbrales():
    """
    'nivel_riesgo' debe ser una de las tres etiquetas esperadas, y debe
    coincidir con la clasificacion que produciria scoring.clasificar_riesgo()
    aplicada directamente sobre proba_default y los dos umbrales
    persistidos -- no se recalcula el umbral aqui, se reutiliza tal cual
    para no duplicar el criterio de decision en el test (mismo patron que
    test_aprobado_es_booleano_y_consistente_con_umbral).

    Ademas, por diseño (ver docstring de clasificar_riesgo), 'bajo' debe
    coincidir exactamente con 'aprobado' -- lo verifica explicitamente
    para dejar constancia de que es una propiedad esperada del sistema,
    no un efecto colateral que pueda romperse sin que ningun test lo note.
    """
    import joblib

    from scoring import clasificar_riesgo

    app = build_graph()
    resultado = app.invoke({"client_data": CLIENTE_EJEMPLO})

    assert resultado["nivel_riesgo"] in {"bajo", "moderado", "alto"}

    umbral_aprobacion = joblib.load("models/decision_threshold.joblib")
    umbral_moderado_alto = joblib.load("models/risk_band_threshold.joblib")
    esperado_nivel = clasificar_riesgo(
        resultado["proba_default"], umbral_aprobacion, umbral_moderado_alto
    )
    assert resultado["nivel_riesgo"] == esperado_nivel, (
        f"nivel_riesgo={resultado['nivel_riesgo']!r} no coincide con la "
        f"clasificacion esperada ({esperado_nivel!r}) para "
        f"proba_default={resultado['proba_default']:.4f}"
    )

    # 'bajo' <=> 'aprobado', por construccion (ambos usan el mismo umbral
    # como frontera inferior).
    assert (resultado["nivel_riesgo"] == "bajo") == resultado["aprobado"], (
        "nivel_riesgo='bajo' deberia coincidir siempre con aprobado=True "
        "(comparten el mismo umbral de aprobacion como frontera)."
    )
