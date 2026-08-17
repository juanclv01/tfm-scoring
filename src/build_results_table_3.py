"""
REPLICA 3: construye la tabla de resultados a partir de
resultados_experimento_3.json (generado por run_experiment_3.py).

No duplica logica: reutiliza build_results_table.construir_tabla(), igual
que build_results_table.py hace con resultados_experimento.json -- solo
cambia el fichero de origen y el fichero de salida.

Uso (identico a build_results_table.py, para esta replica):
    python build_results_table_3.py
"""
from build_results_table import construir_tabla

RESULTADOS_FILE = "resultados_experimento_3.json"
TABLA_FILE = "tabla_resultados_3.md"

if __name__ == "__main__":
    tabla_md = construir_tabla(checkpoint_file=RESULTADOS_FILE)
    with open(TABLA_FILE, "w", encoding="utf-8") as fh:
        fh.write(tabla_md + "\n")
    print(f"\nTabla de la replica 2 guardada en {TABLA_FILE}\n")
    print(tabla_md)
