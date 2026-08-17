"""
Lee un fichero de checkpoint (por defecto RESULTS_CHECKPOINT_FILE) y
construye la tabla comparativa H x B (media +- desviacion estandar sobre
las 20 instancias, por dimension), mas la columna de Puntuacion total
ponderada.

Genera:
  - tabla_resultados.md   (tabla en Markdown, igual estructura que la imagen)
  - imprime tambien un aviso por cada configuracion incompleta (< 20
    instancias), para que sepas cuanto queda por ejecutar.

REUTILIZACION: calcular_medias_por_config() esta separada de construir_tabla()
a proposito -- es la funcion que build_results_table_2.py, _3.py y
build_results_table_final.py importan para leer las medias/n de cualquier
fichero de checkpoint (una replica distinta) sin duplicar la logica de
agregacion aqui en cuatro sitios.
"""
import json

import numpy as np

from experiment_config import (
    CONFIGURACIONES_HB, config_key, RESULTS_CHECKPOINT_FILE, RESULTS_TABLE_FILE,
    N_TEST_INSTANCES, PESOS_DIMENSIONES, ESCALAS_MAX_DIMENSIONES,
)

DIMENSIONES = ["accuracy", "completeness", "fluency", "conciseness", "gdpr"]
NOMBRES_COLUMNA = {
    "accuracy": "Exactitud (accuracy)",
    "completeness": "Completeness (completitud)",
    "fluency": "Fluidez (fluency)",
    "conciseness": "Concisión (conciseness)",
    "gdpr": "Art. 22 GDPR",
}


def puntuacion_total(scores: dict) -> float:
    """Suma ponderada de las 5 dimensiones, cada una normalizada a [0,1]
    dividiendo por su escala maxima nativa. Ver PESOS_DIMENSIONES en
    experiment_config.py -- pesos ya confirmados por Juan (prioriza
    accuracy y GDPR sobre fluency/conciseness)."""
    total = 0.0
    for dim in DIMENSIONES:
        normalizado = scores[dim] / ESCALAS_MAX_DIMENSIONES[dim]
        total += PESOS_DIMENSIONES[dim] * normalizado
    return total


def cargar_checkpoint(checkpoint_file: str = RESULTS_CHECKPOINT_FILE) -> dict:
    with open(checkpoint_file, "r", encoding="utf-8") as fh:
        return json.load(fh)


def calcular_medias_por_config(checkpoint_file: str = RESULTS_CHECKPOINT_FILE) -> dict:
    """
    Devuelve, para cada configuracion (H,B), la media y el tamano de
    muestra de cada dimension (+ la puntuacion total), a partir de UN
    fichero de checkpoint concreto. No formatea nada -- es la pieza
    reutilizable que consumen tanto construir_tabla() (esta misma replica)
    como build_results_table_final.py (combinando las 3 replicas).

    Estructura devuelta:
        {
          "H0_B0": {
            "accuracy": {"media": 0.95, "n": 20},
            ...,
            "total": {"media": 0.812, "n": 20},
          },
          ...
        }
    Si una configuracion no tiene ninguna instancia en este checkpoint,
    no aparece en el dict devuelto (en vez de aparecer con n=0).
    """
    try:
        checkpoint = cargar_checkpoint(checkpoint_file)
    except FileNotFoundError:
        checkpoint = {}

    resultado = {}
    for h, b in CONFIGURACIONES_HB:
        key = config_key(h, b)
        resultados_config = checkpoint.get(key, {})
        n = len(resultados_config)
        if n == 0:
            continue

        valores_por_dim = {d: [] for d in DIMENSIONES}
        totales = []
        for inst_result in resultados_config.values():
            for d in DIMENSIONES:
                valores_por_dim[d].append(inst_result[d])
            totales.append(puntuacion_total(inst_result))

        entrada = {}
        for d in DIMENSIONES:
            arr = np.array(valores_por_dim[d], dtype=float)
            entrada[d] = {"media": float(arr.mean()), "n": n}
        entrada["total"] = {"media": float(np.mean(totales)), "n": n}
        resultado[key] = entrada

    return resultado


def construir_tabla(checkpoint_file: str = RESULTS_CHECKPOINT_FILE) -> str:
    medias_por_config = calcular_medias_por_config(checkpoint_file)

    encabezado = ["H", "B"] + [NOMBRES_COLUMNA[d] for d in DIMENSIONES] + ["Puntuación total"]
    filas = [encabezado, ["---"] * len(encabezado)]

    filas_incompletas = []  # (key, n) para el resumen final

    # Se necesita tambien la desviacion estandar por dimension para la
    # tabla individual de una replica (calcular_medias_por_config() solo
    # da la media, ya que build_results_table_final.py no la necesita a
    # ese nivel -- combina medias de replicas, no instancias sueltas).
    # Por eso aqui se recarga el checkpoint una vez mas para sacar tambien
    # la desviacion estandar por dimension de ESTA replica en concreto.
    try:
        checkpoint = cargar_checkpoint(checkpoint_file)
    except FileNotFoundError:
        checkpoint = {}

    for h, b in CONFIGURACIONES_HB:
        key = config_key(h, b)
        resultados_config = checkpoint.get(key, {})
        n = len(resultados_config)

        if n < N_TEST_INSTANCES:
            filas_incompletas.append((key, n))
            print(f"AVISO: {key} tiene {n}/{N_TEST_INSTANCES} instancias completadas -- "
                  f"la fila se calculara solo con las disponibles (o se dejara en blanco "
                  f"si n=0). Ejecuta run_experiment.py para completarla.")

        fila = [str(h), str(b)]
        if n == 0:
            fila += ["—"] * (len(DIMENSIONES) + 1)
            filas.append(fila)
            continue

        valores_por_dim = {d: [] for d in DIMENSIONES}
        totales = []
        for inst_result in resultados_config.values():
            for d in DIMENSIONES:
                valores_por_dim[d].append(inst_result[d])
            totales.append(puntuacion_total(inst_result))

        for d in DIMENSIONES:
            arr = np.array(valores_por_dim[d], dtype=float)
            fila.append(f"{arr.mean():.2f} ± {arr.std():.2f} (n={n})")

        arr_total = np.array(totales, dtype=float)
        fila.append(f"{arr_total.mean():.3f} ± {arr_total.std():.3f} (n={n})")

        filas.append(fila)

    lineas_md = ["| " + " | ".join(fila) + " |" for fila in filas]
    tabla = "\n".join(lineas_md)

    # Leyenda de completitud escrita DENTRO del propio fichero -- no solo por
    # consola -- para que sea legible por si sola sin haber visto la
    # ejecucion que la genero (importante dado que el proceso experimental
    # puede tardar mucho y construirse la tabla en varias sesiones distintas
    # con avance parcial en distintas configuraciones).
    total_configs = len(CONFIGURACIONES_HB)
    completas = total_configs - len(filas_incompletas)
    resumen = [
        "",
        f"**Fichero de origen**: `{checkpoint_file}`",
        f"**Estado de completitud**: {completas}/{total_configs} configuraciones con "
        f"las {N_TEST_INSTANCES} instancias completas.",
    ]
    if filas_incompletas:
        resumen.append(
            "Configuraciones incompletas (el `(n=...)` de cada celda indica cuantas "
            "instancias entraron en esa media/desviacion -- los resultados son validos "
            "pero calculados sobre una muestra menor a la prevista):"
        )
        for key, n in filas_incompletas:
            resumen.append(f"- `{key}`: {n}/{N_TEST_INSTANCES} instancias")
    else:
        resumen.append("Todas las configuraciones estan completas.")

    return tabla + "\n" + "\n".join(resumen)


if __name__ == "__main__":
    tabla_md = construir_tabla()
    with open(RESULTS_TABLE_FILE, "w", encoding="utf-8") as fh:
        fh.write(tabla_md + "\n")
    print(f"\nTabla guardada en {RESULTS_TABLE_FILE}\n")
    print(tabla_md)
