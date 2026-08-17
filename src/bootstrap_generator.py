"""
Genera los ejemplares BOOTSTRAPPED (B) de Explingo: narrativas sinteticas
generadas por el propio NARRATOR (en modo H=0, B=0, zero-shot) sobre
instancias del conjunto de TRAIN (nunca del test set de 20 instancias usado
en la experimentacion), que se aceptan como ejemplar solo si el GRADER las
valida en accuracy y completeness (Seccion 11 del contexto del nodo 2 --
metodologia identica a la de Zytek et al., que tampoco filtra por GDPR en
esta fase: GDPR es la dimension añadida del TFM, y debe quedar libre de
observar en la rejilla H x B si emerge por imitacion igual que el resto,
no fijarse de antemano por el propio criterio de aceptacion del pool B).

Los B=1/3/5 exemplars son ANIDADOS (ver experiment_config): se genera un
pool candidato diverso, se evaluan en orden hasta acumular 5 aceptados, y
B=1/B=3 son prefijos de esa misma lista de 5.

Checkpointed: cada candidato procesado (aceptado o rechazado) se persiste
en BOOTSTRAP_CANDIDATE_POOL_FILE inmediatamente, de forma que una ejecucion
interrumpida por limite de cuota puede reanudarse sin repetir candidatos ya
evaluados.
"""
import json
import os

# IMPORTANTE -- orden de imports deliberado: joblib (y numpy, que arrastra
# consigo) DEBEN importarse antes que dspy. dspy instala un gancho de
# "lazy import" global sobre ciertas librerias pesadas (numpy incluida); si
# numpy se importa por primera vez DESPUES de que ese gancho ya este
# activo, numpy queda parcialmente inicializado dos veces y falla con
# "TypeError: data type 'bool' not understood". Importando joblib (y por
# tanto numpy) primero, numpy queda completamente cargado en sys.modules
# antes de que dspy tenga ocasion de interceptarlo. No reordenar estas
# lineas al editar este fichero.
import joblib

from data_loader import load_german_credit, split_data
from generate_shap_examples import seleccionar_instancias_diversas
from select_test_instances import evaluar_estado_completo, cargar_o_seleccionar_instancias

import dspy
from narrativas_hand_written import CONTEXT, EXPLANATION_FORMAT
from dspy_modules import (
    NarratorSignature, AccuracySignature, CompletenessSignature,
    construir_lm_narrator, construir_lm_grader, llamar_con_pausa, limpiar_narrativa,
)
from experiment_config import (
    MAX_B, BOOTSTRAP_ACCURACY_MIN, BOOTSTRAP_COMPLETENESS_MIN,
    BOOTSTRAP_CANDIDATE_POOL_FILE, BOOTSTRAP_EXAMPLES_FILE,
    NARRATOR_SECONDS_BETWEEN_CALLS, GRADER_SECONDS_BETWEEN_CALLS,
    N_TEST_INSTANCES, TEST_INSTANCES_SEED,
)

N_CANDIDATOS_POOL = 20  # tamano del pool diverso a explorar; se para en cuanto
                        # se acumulan MAX_B aceptados, no hace falta procesarlo entero

_narrator_predict = dspy.Predict(NarratorSignature)
_accuracy_predict = dspy.Predict(AccuracySignature)
_completeness_predict = dspy.Predict(CompletenessSignature)


def _formatear_explanation(top_features: list) -> str:
    return "\n".join(
        f"({f['feature_name']}, {f['feature_value']}, {f['shap_value']:+.1f} pts)"
        for f in top_features
    )


def _cargar_pool() -> dict:
    if os.path.exists(BOOTSTRAP_CANDIDATE_POOL_FILE):
        with open(BOOTSTRAP_CANDIDATE_POOL_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def _guardar_pool(pool: dict) -> None:
    with open(BOOTSTRAP_CANDIDATE_POOL_FILE, "w", encoding="utf-8") as fh:
        json.dump(pool, fh, ensure_ascii=False, indent=2)


def _seleccionar_candidatos_train(exclude_test_indices: set) -> list:
    """Selecciona un pool diverso de instancias de TRAIN (nunca de test),
    reutilizando la misma logica de diversidad que ya usamos para las 30
    instancias de test y para las 5 H. exclude_test_indices se pasa solo
    por claridad -- train y test ya son particiones disjuntas por
    construccion de split_data(), no hay riesgo real de solapamiento."""
    model = joblib.load("models/xgb_scoring_pipeline.joblib")
    df = load_german_credit("data/german-credit-data/german.data")
    X_train, _, _, _ = split_data(df)

    seleccion = seleccionar_instancias_diversas(
        X_train, model, n=N_CANDIDATOS_POOL, seed=TEST_INSTANCES_SEED
    )
    return [
        {"candidato_id": f"train_{idx}", "client_data": estado["client_data"]}
        for idx, estado in seleccion
    ]


def generar_ejemplares_bootstrapped(max_llamadas: int = None) -> list:
    """
    Ejecuta (o reanuda) la generacion de ejemplares bootstrapped.

    max_llamadas: si se especifica, para tras aproximadamente ese numero de
                  llamadas LLM en esta invocacion (para repartir el gasto de
                  cuota entre sesiones). Cada candidato consume hasta 3
                  llamadas (narrator + accuracy + completeness).

    Devuelve la lista final de hasta MAX_B ejemplares aceptados (dicts con
    explanation/narrative/ground_truth), ya persistida en
    BOOTSTRAP_EXAMPLES_FILE si se completa el proceso.
    """
    pool = _cargar_pool()
    llamadas_hechas = 0

    if not pool:
        test_instances = cargar_o_seleccionar_instancias()
        test_idx = {i["indice_test"] for i in test_instances}
        candidatos = _seleccionar_candidatos_train(exclude_test_indices=test_idx)
        for c in candidatos:
            pool[c["candidato_id"]] = {"client_data": c["client_data"], "estado": "pendiente"}
        _guardar_pool(pool)
        print(f"Pool de {len(candidatos)} candidatos de train creado en {BOOTSTRAP_CANDIDATE_POOL_FILE}")

    aceptados = [v for v in pool.values() if v.get("estado") == "aceptado"]
    if len(aceptados) >= MAX_B:
        print(f"Ya hay {len(aceptados)} candidatos aceptados (>= MAX_B={MAX_B}). Nada que hacer.")
        return _guardar_ejemplares_finales(pool)

    lm_narrator = construir_lm_narrator()
    lm_grader = construir_lm_grader()

    for candidato_id, info in pool.items():
        aceptados = [v for v in pool.values() if v.get("estado") == "aceptado"]
        if len(aceptados) >= MAX_B:
            break
        if info.get("estado") not in (None, "pendiente"):
            continue  # ya procesado (aceptado o rechazado) en una sesion anterior
        if max_llamadas is not None and llamadas_hechas >= max_llamadas:
            print(f"Alcanzado el limite de {max_llamadas} llamadas en esta sesion. "
                  f"Reanuda mas tarde ejecutando este script de nuevo.")
            break

        try:
            estado = evaluar_estado_completo(info["client_data"])
            explanation = _formatear_explanation(estado["top_features"])
            ground_truth = (
                f"Decision: {'APROBADA' if estado['aprobado'] else 'RECHAZADA'}\n"
                f"Risk level: {estado['nivel_riesgo']}\n"
                f"Score: {estado['score']}/1000"
            )

            narrator_out = llamar_con_pausa(
                _narrator_predict, lm_narrator, NARRATOR_SECONDS_BETWEEN_CALLS,
                context=CONTEXT,
                decision="APROBADA" if estado["aprobado"] else "RECHAZADA",
                risk_level=estado["nivel_riesgo"],
                score=estado["score"],
                explanation=explanation,
                explanation_format=EXPLANATION_FORMAT,
            )
            narrative = limpiar_narrativa(narrator_out.narrative)
            llamadas_hechas += 1

            acc_out = llamar_con_pausa(
                _accuracy_predict, lm_grader, GRADER_SECONDS_BETWEEN_CALLS,
                ground_truth=ground_truth, explanation=explanation,
                explanation_format=EXPLANATION_FORMAT, narrative=narrative,
            )
            accuracy = int(acc_out.assessment)
            llamadas_hechas += 1

            comp_out = llamar_con_pausa(
                _completeness_predict, lm_grader, GRADER_SECONDS_BETWEEN_CALLS,
                explanation=explanation, explanation_format=EXPLANATION_FORMAT,
                narrative=narrative,
            )
            completeness = int(comp_out.assessment)
            llamadas_hechas += 1

            pasa = (accuracy >= BOOTSTRAP_ACCURACY_MIN) and (completeness >= BOOTSTRAP_COMPLETENESS_MIN)

            info.update({
                "estado": "aceptado" if pasa else "rechazado",
                "narrative": narrative,
                "explanation": explanation,
                "ground_truth": ground_truth,
                "decision": "APROBADA" if estado["aprobado"] else "RECHAZADA",
                "risk_level": estado["nivel_riesgo"],
                "score": estado["score"],
                "accuracy": accuracy,
                "completeness": completeness,
            })
            _guardar_pool(pool)  # checkpoint inmediato tras cada candidato

            print(f"  {candidato_id}: accuracy={accuracy} completeness={completeness} -> {info['estado']}")

        except Exception as exc:
            print(f"  {candidato_id}: ERROR ({exc}) -- se deja como pendiente, se reintentara en la proxima sesion")
            _guardar_pool(pool)
            continue

    return _guardar_ejemplares_finales(pool)


def _guardar_ejemplares_finales(pool: dict) -> list:
    aceptados = [v for v in pool.values() if v.get("estado") == "aceptado"][:MAX_B]
    with open(BOOTSTRAP_EXAMPLES_FILE, "w", encoding="utf-8") as fh:
        json.dump(aceptados, fh, ensure_ascii=False, indent=2)
    print(f"{len(aceptados)}/{MAX_B} ejemplares bootstrapped aceptados, guardados en {BOOTSTRAP_EXAMPLES_FILE}")
    return aceptados


def obtener_demos_bootstrapped(b: int) -> list:
    """Primeros b ejemplares bootstrapped ya aceptados (deben haberse
    generado antes con generar_ejemplares_bootstrapped()), en formato
    dspy.Example listos para predict.demos."""
    if not os.path.exists(BOOTSTRAP_EXAMPLES_FILE):
        raise RuntimeError(
            f"{BOOTSTRAP_EXAMPLES_FILE} no existe todavia. Ejecuta primero "
            f"generar_ejemplares_bootstrapped() (python bootstrap_generator.py)."
        )
    with open(BOOTSTRAP_EXAMPLES_FILE, "r", encoding="utf-8") as fh:
        ejemplares = json.load(fh)
    if len(ejemplares) < b:
        raise RuntimeError(
            f"Solo hay {len(ejemplares)} ejemplares bootstrapped aceptados, "
            f"se necesitan {b}. Reanuda generar_ejemplares_bootstrapped()."
        )

    demos = []
    for item in ejemplares[:b]:
        ex = dspy.Example(
            context=CONTEXT,
            decision=item["decision"],
            risk_level=item["risk_level"],
            score=item["score"],
            explanation=item["explanation"],
            explanation_format=EXPLANATION_FORMAT,
            narrative=item["narrative"],
        ).with_inputs("context", "decision", "risk_level", "score", "explanation", "explanation_format")
        demos.append(ex)
    return demos


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-llamadas", type=int, default=None,
                         help="tope de llamadas LLM en esta sesion (para repartir cuota)")
    args = parser.parse_args()
    generar_ejemplares_bootstrapped(max_llamadas=args.max_llamadas)
