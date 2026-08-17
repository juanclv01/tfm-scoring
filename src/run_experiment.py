"""
Orquesta la experimentacion completa: para cada configuracion (H, B) de
experiment_config.CONFIGURACIONES_HB, genera una narrativa por cada una de
las 20 instancias fijas y la evalua en las 5 dimensiones.

DISENADO PARA EJECUTARSE EN VARIAS SESIONES (free tier limitado):
- Cada (config, instancia) procesada se guarda INMEDIATAMENTE en
  RESULTS_CHECKPOINT_FILE. Si el proceso se interrumpe (por cuota agotada,
  error de red, o Ctrl+C), la siguiente ejecucion retoma justo donde se
  quedo -- nunca se recomputa una narrativa/evaluacion ya hecha.
- --max-llamadas limita cuantas llamadas LLM (narrator + grader) se hacen
  en ESTA invocacion del script. Cada instancia cuesta 1 llamada narrator +
  4 llamadas grader = 5 llamadas. Con esto puedes, por ejemplo, ejecutar
  `python run_experiment.py --max-llamadas 100` cada dia hasta completar
  las 11 configuraciones x 20 instancias = 220 narrativas (~1100 llamadas totales, ~880 del GRADER).
- --configs permite restringir a un subconjunto de configuraciones
  concretas en esta sesion, p.ej. --configs H0_B0 H1_B0
- evaluar_estado_completo() (score/SHAP/decision) se precomputa UNA VEZ
  por instancia en _construir_cache_instancias(), no una vez por cada
  combinacion (instancia, configuracion) -- ese calculo no depende de H
  ni B, solo del client_data.

Uso tipico (repetido dia a dia hasta completar):
    python run_experiment.py --max-llamadas 100
    python run_experiment.py --max-llamadas 100
    ...
    python build_results_table.py     # cuando este todo completo
"""
import argparse
import hashlib
import json
import os

# IMPORTANTE -- mismo motivo que en bootstrap_generator.py: select_test_instances
# arrastra joblib/numpy y debe importarse ANTES que dspy, o numpy se rompe
# con "TypeError: data type 'bool' not understood" por el gancho de lazy
# import de dspy. No reordenar estas lineas.
from select_test_instances import cargar_o_seleccionar_instancias, evaluar_estado_completo

import dspy
from narrativas_hand_written import CONTEXT, EXPLANATION_FORMAT, obtener_demos_narrator
from bootstrap_generator import obtener_demos_bootstrapped
from dspy_modules import NarratorSignature, construir_lm_narrator, construir_lm_grader, llamar_con_pausa, limpiar_narrativa
from metrics import evaluar_narrativa
from experiment_config import (
    CONFIGURACIONES_HB, config_key, RESULTS_CHECKPOINT_FILE,
    NARRATOR_SECONDS_BETWEEN_CALLS,
)

_narrator_predict = dspy.Predict(NarratorSignature)


def _hash_client_data(client_data: dict) -> str:
    """Huella corta y estable de un client_data, para detectar si una
    entrada de checkpoint sigue correspondiendo al mismo cliente. Se
    recalcula cada vez que se selecciona/carga el test set y se compara
    contra la huella guardada en el checkpoint -- si no coincide, la
    entrada se trata como obsoleta y se recalcula, en vez de confiar
    ciegamente en que el numero de indice_test significa lo mismo que
    la ultima vez (no lo significa si test_instances_20.json se regenero)."""
    serializado = json.dumps(client_data, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(serializado.encode("utf-8")).hexdigest()[:12]


def _formatear_explanation(top_features: list) -> str:
    return "\n".join(
        f"({f['feature_name']}, {f['feature_value']}, {f['shap_value']:+.1f} pts)"
        for f in top_features
    )


def _cargar_checkpoint(checkpoint_file: str = RESULTS_CHECKPOINT_FILE) -> dict:
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def _guardar_checkpoint(checkpoint: dict, checkpoint_file: str = RESULTS_CHECKPOINT_FILE) -> None:
    with open(checkpoint_file, "w", encoding="utf-8") as fh:
        json.dump(checkpoint, fh, ensure_ascii=False, indent=2)


def _exemplars_para_fluency(h: int) -> str:
    """Texto de las narrativas H usadas como referencia de estilo. Cadena
    vacia si H=0 (ver nota en FluencySignature: el GRADER debe juzgar
    fluidez general en ese caso, no comparar contra un vacio)."""
    from narrativas_hand_written import NARRATIVAS_HAND_WRITTEN
    if h == 0:
        return ""
    return "\n\n---\n\n".join(item["narrative"] for item in NARRATIVAS_HAND_WRITTEN[:h])


def _construir_cache_instancias(instancias: list) -> dict:
    """
    Precomputa evaluar_estado_completo() (score, SHAP, decision, nivel de
    riesgo) UNA SOLA VEZ por instancia, en vez de una vez por cada
    combinacion (instancia, configuracion H/B) -- el resultado de esta
    funcion no depende de H ni B en absoluto (solo del client_data), asi
    que recalcularlo 11 veces por instancia (una por fila de la tabla) era
    trabajo redundante. Con 20 instancias, esto reduce 220 llamadas a
    evaluar_estado_completo() a solo 20.

    Se calcula fuera del bucle principal y sin try/except individual a
    proposito: un fallo aqui es un problema determinista de datos/entorno
    (modelo o background de SHAP ausente, client_data malformado), no un
    fallo transitorio de red -- debe interrumpir la ejecucion con un error
    claro en vez de reintentarse silenciosamente en cada sesion futura.
    """
    cache = {}
    for inst in instancias:
        inst_id = str(inst["indice_test"])
        estado = evaluar_estado_completo(inst["client_data"])
        cache[inst_id] = {
            "estado": estado,
            "explanation": _formatear_explanation(estado["top_features"]),
            "ground_truth": (
                f"Decision: {'APROBADA' if estado['aprobado'] else 'RECHAZADA'}\n"
                f"Risk level: {estado['nivel_riesgo']}\n"
                f"Score: {estado['score']}/1000"
            ),
            "num_features": len(estado["top_features"]),
            "client_data_hash": _hash_client_data(inst["client_data"]),
        }
    return cache


def ejecutar(configs_a_procesar: list, max_llamadas: int = None,
             checkpoint_file: str = RESULTS_CHECKPOINT_FILE) -> None:
    checkpoint = _cargar_checkpoint(checkpoint_file)
    instancias = cargar_o_seleccionar_instancias()
    cache_instancias = _construir_cache_instancias(instancias)

    lm_narrator = construir_lm_narrator()
    lm_grader = construir_lm_grader()

    llamadas_hechas = 0
    LLAMADAS_POR_INSTANCIA = 5  # 1 narrator + 4 grader (accuracy, completeness, fluency, gdpr)

    for h, b in configs_a_procesar:
        key = config_key(h, b)
        checkpoint.setdefault(key, {})

        # Demos fijos para esta configuracion (constante durante las 20 instancias)
        demos_h = obtener_demos_narrator(h)
        demos_b = obtener_demos_bootstrapped(b) if b > 0 else []
        _narrator_predict.demos = demos_h + demos_b
        exemplars_fluency = _exemplars_para_fluency(h)

        print(f"\n=== Configuracion {key} ({len(demos_h)} H + {len(demos_b)} B demos) "
              f"[checkpoint: {checkpoint_file}] ===")

        for inst in instancias:
            inst_id = str(inst["indice_test"])
            datos = cache_instancias[inst_id]
            huella_actual = datos["client_data_hash"]
            entrada_previa = checkpoint[key].get(inst_id)
            if entrada_previa is not None:
                if entrada_previa.get("client_data_hash") == huella_actual:
                    continue  # ya procesado en una sesion anterior, mismo cliente
                print(f"  instancia {inst_id}: AVISO -- el resultado guardado no "
                      f"corresponde al cliente actual (huella distinta, probablemente "
                      f"por una regeneracion de test_instances_20.json). Se recalcula.")
            if max_llamadas is not None and llamadas_hechas + LLAMADAS_POR_INSTANCIA > max_llamadas:
                print(f"Presupuesto de {max_llamadas} llamadas agotado en esta sesion. "
                      f"Reanuda mas tarde ejecutando este mismo script de nuevo.")
                _guardar_checkpoint(checkpoint, checkpoint_file)
                return

            try:
                estado = datos["estado"]
                explanation = datos["explanation"]
                ground_truth = datos["ground_truth"]

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

                scores = evaluar_narrativa(
                    narrative=narrative,
                    ground_truth=ground_truth,
                    explanation=explanation,
                    explanation_format=EXPLANATION_FORMAT,
                    exemplars=exemplars_fluency,
                    num_features=datos["num_features"],
                    grader_lm=lm_grader,
                )

                checkpoint[key][inst_id] = {
                    "narrative": narrative,
                    "client_data_hash": huella_actual,
                    **scores,
                }
                llamadas_hechas += LLAMADAS_POR_INSTANCIA
                _guardar_checkpoint(checkpoint, checkpoint_file)  # checkpoint tras CADA instancia

                print(f"  instancia {inst_id}: {scores}")

            except Exception as exc:
                print(f"  instancia {inst_id}: ERROR ({exc}) -- se reintentara en la proxima sesion")
                _guardar_checkpoint(checkpoint, checkpoint_file)
                continue

    print("\nTodas las configuraciones solicitadas estan completas (o se ha agotado el presupuesto).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-llamadas", type=int, default=None,
                         help="tope de llamadas LLM (narrator+grader) en esta sesion")
    parser.add_argument("--configs", nargs="*", default=None,
                         help="subconjunto de configs a procesar, p.ej. H0_B0 H1_B0 "
                              "(por defecto: todas las de CONFIGURACIONES_HB)")
    parser.add_argument("--checkpoint-file", type=str, default=RESULTS_CHECKPOINT_FILE,
                         help="fichero de checkpoint a usar (por defecto: "
                              "resultados_experimento.json). Util para replicas "
                              "independientes -- ver build_results_table_2.py / _3.py")
    args = parser.parse_args()

    if args.configs:
        pares = []
        for c in args.configs:
            h_str, b_str = c.replace("H", "").split("_B")
            pares.append((int(h_str), int(b_str)))
    else:
        pares = CONFIGURACIONES_HB

    ejecutar(pares, max_llamadas=args.max_llamadas, checkpoint_file=args.checkpoint_file)
