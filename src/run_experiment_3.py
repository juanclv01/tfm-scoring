"""
REPLICA 3 de la experimentacion, para validacion estadistica de la
robustez de los resultados frente a la aleatoriedad de los LLM
(NARRATOR_TEMPERATURE=0.7 introduce variacion entre ejecuciones incluso
con la misma configuracion H/B y la misma instancia).

No duplica logica: reutiliza run_experiment.ejecutar() (la funcion real
que llama al NARRATOR/GRADER, con checkpoint y --max-llamadas), apuntando
a RESULTADOS_FILE en vez de al de la replica 1. Cualquier correccion
futura de metodologia (prompts, formula de conciseness, pesos, etc.) en
run_experiment.py se aplica automaticamente a esta replica tambien.

RESUMIBLE igual que run_experiment.py: si se interrumpe (Ctrl+C, cierre de
VSCode, cuota agotada), la siguiente ejecucion retoma justo donde se
quedo, sin recomputar nada ya guardado en RESULTADOS_FILE.

Uso (identico a run_experiment.py):
    python run_experiment_3.py --max-llamadas 100
    python run_experiment_3.py --max-llamadas 100   # repetir hasta completar
    python run_experiment_3.py --configs H1_B0 --max-llamadas 100

Cuando quieras ver la tabla de esta replica: python build_results_table_3.py
"""
import argparse

from run_experiment import ejecutar
from experiment_config import CONFIGURACIONES_HB

RESULTADOS_FILE = "resultados_experimento_3.json"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-llamadas", type=int, default=None,
                         help="tope de llamadas LLM (narrator+grader) en esta sesion")
    parser.add_argument("--configs", nargs="*", default=None,
                         help="subconjunto de configs a procesar, p.ej. H0_B0 H1_B0 "
                              "(por defecto: todas las de CONFIGURACIONES_HB)")
    args = parser.parse_args()

    if args.configs:
        pares = []
        for c in args.configs:
            h_str, b_str = c.replace("H", "").split("_B")
            pares.append((int(h_str), int(b_str)))
    else:
        pares = CONFIGURACIONES_HB

    ejecutar(pares, max_llamadas=args.max_llamadas, checkpoint_file=RESULTADOS_FILE)
