"""
Utilidades compartidas por run_experiment.py, bootstrap_generator.py y la
aplicacion (app_backend.py). Antes esta logica vivia duplicada en cada
script por separado -- se centraliza aqui para que una correccion futura
(p.ej. el formato de ground_truth cuando se anadio el campo Score) solo
haya que hacerla en un sitio.
"""
import hashlib
import json


def hash_client_data(client_data: dict) -> str:
    """Huella corta y estable de un client_data. Se usa para:
    (a) detectar si una entrada de checkpoint sigue correspondiendo al
        mismo cliente (ver run_experiment.py),
    (b) como identificador de trazabilidad no identificativo en la app
        (en vez de mostrar el nombre del solicitante)."""
    serializado = json.dumps(client_data, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(serializado.encode("utf-8")).hexdigest()[:12]


def formatear_explanation(top_features: list) -> str:
    """Formato NARRATOR estandar: (nombre, valor, shap en pts con signo)."""
    return "\n".join(
        f"({f['feature_name']}, {f['feature_value']}, {f['shap_value']:+.1f} pts)"
        for f in top_features
    )


def construir_ground_truth(estado: dict) -> str:
    """Ground truth usado por el GRADER (accuracy/gdpr): decision, nivel
    de riesgo y score, tal como los calcula node_scoring/node_explainability
    -- nunca inventado ni recalculado aqui."""
    return (
        f"Decision: {'APROBADA' if estado['aprobado'] else 'RECHAZADA'}\n"
        f"Risk level: {estado['nivel_riesgo']}\n"
        f"Score: {estado['score']}/1000"
    )
