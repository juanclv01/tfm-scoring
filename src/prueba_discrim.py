"""
Prueba de VALIDEZ DISCRIMINATIVA del GRADER: comprueba que un GRADER que
siempre da notas altas no es lo mismo que un GRADER que funciona bien --
si nunca se le da una narrativa mala, no hay forma de distinguir ambos
casos. Este script construye una narrativa deliberadamente defectuosa,
con fallos conocidos y verificables a mano, y confirma que el GRADER los
detecta y puntua bajo.

Fallos deliberados de la narrativa de prueba:
  1. DIRECCION SHAP INCORRECTA: la Explanation dice que "sin cuenta
     corriente" aporta +75.0 pts (SUBE el score), pero la narrativa
     afirma que "ha reducido la puntuacion" (direccion invertida).
  2. SCORE AUSENTE: la narrativa nunca menciona el numero 812/1000,
     pese a ser obligatorio segun el diseno actual del NARRATOR.
  3. INCOMPLETITUD: la Explanation tiene 5 factores, la narrativa solo
     menciona 3 (omite "Historial crediticio" y "Situacion laboral"
     por completo).
  4. SIN CONCORDANCIA CON EL ART. 22 RGPD: no hay ninguna mencion al
     derecho a impugnacion / revision humana en todo el texto.

USA EXACTAMENTE LA MISMA CONFIGURACION que la experimentacion real:
  - construir_lm_grader() de dspy_modules.py (mismo modelo, mismo
    proveedor, mismo max_tokens, mismo extra_body)
  - evaluar_narrativa() de metrics.py (misma logica de evaluacion,
    sin reimplementar nada aqui)

Uso:
    python prueba_discrim.py
"""
import joblib  # numpy antes que dspy -- ver nota en run_experiment.py

from dspy_modules import construir_lm_grader
from metrics import evaluar_narrativa
from narrativas_hand_written import CONTEXT, EXPLANATION_FORMAT


# ---------------------------------------------------------------------
# Datos de la instancia de prueba (la "verdad" contra la que se compara)
# ---------------------------------------------------------------------
SCORE_REAL = 812
DECISION_REAL = "APROBADA"
RISK_LEVEL_REAL = "bajo"

EXPLANATION = (
    "(Estado de la cuenta corriente, sin cuenta corriente, +75.0 pts)\n"
    "(Otros planes de pago activos, sin planes de pago adicionales, +9.0 pts)\n"
    "(Historial crediticio, cuenta crítica o créditos existentes en otras entidades, +8.0 pts)\n"
    "(Duración del préstamo (meses), 12 meses, +6.0 pts)\n"
    "(Situación laboral, empleado desde hace 7 años o más, +3.0 pts)"
)
NUM_FEATURES = 5

GROUND_TRUTH = (
    f"Decision: {DECISION_REAL}\n"
    f"Risk level: {RISK_LEVEL_REAL}\n"
    f"Score: {SCORE_REAL}/1000"
)

# ---------------------------------------------------------------------
# Narrativa DELIBERADAMENTE MALA (ver los 4 fallos documentados arriba)
# ---------------------------------------------------------------------
NARRATIVA_MALA = (
    "La solicitud ha sido aprobada, con un nivel de riesgo bajo. El hecho de no "
    "disponer de cuenta corriente ha reducido la puntuación en 75 puntos. Además, "
    "la ausencia de planes de pago adicionales ha sumado 9 puntos, y la duración "
    "del préstamo de 12 meses ha aportado 6 puntos adicionales."
)

# Narrativa de referencia para fluency (una H real, para que la comparacion de
# estilo tenga sentido -- no es parte de los 4 fallos que se estan probando,
# es solo para que fluency tenga contra que comparar en vez de evaluarse vacia).
EXEMPLAR_FLUENCY = (
    "Su solicitud de préstamo personal ha sido aprobada con una puntuación "
    "crediticia de 812 sobre 1000, lo que la clasifica en la categoría de "
    "riesgo bajo, superando de esta forma nuestros criterios de aceptación."
)


def main():
    print("=" * 70)
    print("PRUEBA DE VALIDEZ DISCRIMINATIVA DEL GRADER")
    print("=" * 70)
    print("\nNarrativa evaluada (deliberadamente mala):\n")
    print(f"  {NARRATIVA_MALA}\n")
    print("Fallos esperados: direccion SHAP invertida, score ausente, "
          "2/5 factores omitidos, sin mencion al art. 22 RGPD.\n")

    lm_grader = construir_lm_grader()  # MISMA config que run_experiment.py/bootstrap_generator.py

    resultado = evaluar_narrativa(
        narrative=NARRATIVA_MALA,
        ground_truth=GROUND_TRUTH,
        explanation=EXPLANATION,
        explanation_format=EXPLANATION_FORMAT,
        exemplars=EXEMPLAR_FLUENCY,
        num_features=NUM_FEATURES,
        grader_lm=lm_grader,
    )

    justificaciones = resultado["justificaciones"]

    print("-" * 70)
    print(f"{'Dimension':<15}{'Puntuacion':<15}{'Esperado':<15}{'Resultado'}")
    print("-" * 70)

    # (dimension, valor obtenido, umbral maximo esperable si el GRADER
    #  discrimina bien, descripcion del umbral)
    comprobaciones = [
        ("accuracy", resultado["accuracy"], 0, "debe ser 0 (direccion invertida + score ausente)"),
        ("completeness", resultado["completeness"], 0, "debe ser 0 (2/5 factores omitidos)"),
        ("gdpr", resultado["gdpr"], 1, "debe ser <=1 (sin mencion al art. 22)"),
    ]

    algun_fallo_no_detectado = False
    for nombre, valor, maximo_esperado, descripcion in comprobaciones:
        ok = valor <= maximo_esperado
        estado = "OK" if ok else "!! NO DETECTADO"
        if not ok:
            algun_fallo_no_detectado = True
        print(f"{nombre:<15}{valor!s:<15}{descripcion:<45}{estado}")

    print(f"{'fluency':<15}{resultado['fluency']!s:<15}"
          f"{'(informativo, no forma parte de los 4 fallos probados)':<45}")
    print(f"{'conciseness':<15}{resultado['conciseness']!s:<15}"
          f"{'(informativo, formula deterministica, no aplica aqui)':<45}")

    print("\n" + "-" * 70)
    print("JUSTIFICACIONES DEL GRADER (solo para ti, nunca se muestran al cliente):")
    print("-" * 70)
    for dim in ["accuracy", "completeness", "fluency", "gdpr"]:
        print(f"\n[{dim}]\n  {justificaciones[dim]}")

    print("\n" + "=" * 70)
    if algun_fallo_no_detectado:
        print("RESULTADO: el GRADER NO detecto correctamente uno o mas fallos "
              "deliberados. Revisa las justificaciones de arriba -- puede "
              "indicar que algun prompt necesita ajustarse.")
    else:
        print("RESULTADO: el GRADER detecto correctamente los fallos deliberados "
              "(puntuaciones bajas donde se esperaban). Validez discriminativa "
              "confirmada con este caso de prueba.")
    print("=" * 70)


if __name__ == "__main__":
    main()
