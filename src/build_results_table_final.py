"""
Combina las 3 replicas independientes de la experimentacion
(resultados_experimento.json, _2.json, _3.json) en una unica tabla final.

CRITERIO DE COMBINACION (importante, confirmar que es el que Juan quiere):
para cada configuracion (H,B) y cada dimension, se toma la MEDIA que cada
replica ya calculo sobre sus propias instancias, y estos 3 numeros (uno
por replica) se promedian entre si, con su propia desviacion estandar.

Es decir: NO se juntan las instancias de las 3 replicas en una sola bolsa
de 60 valores -- se promedian las 3 medias ya calculadas. Esto responde a
la pregunta "?cuanto varia el resultado de una ejecucion completa a otra?"
(robustez frente al componente estocastico del LLM, temperature=0.7), que
es una pregunta distinta (y la que Juan pidio) de "?cual es la media
combinando todas las observaciones individuales?" -- esa segunda pregunta
tendria mas peso estadistico pero diluiria precisamente la senal que se
quiere medir (variabilidad ENTRE ejecuciones completas).

Solo se combinan las configuraciones que tengan datos en las 3 replicas
Y con el mismo n en las 3 (mismo grado de completitud) -- si una replica
tiene una fila incompleta o ausente, esa fila se omite de la tabla final
con un aviso, en vez de promediar medias calculadas sobre tamanos de
muestra distintos entre si.

Uso (una vez las 3 replicas tengan al menos las filas que interesen, p.ej.
H1_B0, generadas):
    python build_results_table_final.py
"""
from build_results_table import (
    calcular_medias_por_config, DIMENSIONES, NOMBRES_COLUMNA,
)
from experiment_config import CONFIGURACIONES_HB, config_key, N_TEST_INSTANCES

FICHEROS_REPLICAS = [
    "resultados_experimento.json",
    "resultados_experimento_2.json",
    "resultados_experimento_3.json",
]
TABLA_FINAL_FILE = "tabla_resultados_final.md"


def combinar() -> str:
    medias_por_replica = [calcular_medias_por_config(f) for f in FICHEROS_REPLICAS]

    encabezado = ["H", "B"] + [NOMBRES_COLUMNA[d] for d in DIMENSIONES] + ["Puntuación total"]
    filas = [encabezado, ["---"] * len(encabezado)]

    filas_omitidas = []

    for h, b in CONFIGURACIONES_HB:
        key = config_key(h, b)

        # Comprobar que las 3 replicas tienen esta configuracion, con el
        # mismo n en las 3 (mismo grado de completitud -- si no, las medias
        # no son comparables entre si sin mas matices).
        entradas = [medias.get(key) for medias in medias_por_replica]
        if any(e is None for e in entradas):
            n_disponibles = [
                (i + 1, medias[key]["accuracy"]["n"]) for i, medias in enumerate(medias_por_replica)
                if key in medias
            ]
            filas_omitidas.append(
                f"`{key}`: falta en {sum(1 for e in entradas if e is None)}/3 replicas "
                f"(presente en: {n_disponibles})"
            )
            continue

        ns = {entradas[i]["accuracy"]["n"] for i in range(3)}
        if len(ns) > 1:
            filas_omitidas.append(
                f"`{key}`: presente en las 3 replicas pero con n distinto entre ellas "
                f"({[entradas[i]['accuracy']['n'] for i in range(3)]}) -- no comparable "
                f"directamente, omitida. Completa las 3 replicas al mismo n antes de combinar."
            )
            continue

        n_por_replica = entradas[0]["accuracy"]["n"]
        fila = [str(h), str(b)]
        for d in DIMENSIONES + ["total"]:
            medias_3 = [entradas[i][d]["media"] for i in range(3)]
            media_final = sum(medias_3) / 3
            desv_final = (sum((m - media_final) ** 2 for m in medias_3) / 3) ** 0.5
            if d == "total":
                fila.append(f"{media_final:.3f} ± {desv_final:.3f} (n={n_por_replica}x3)")
            else:
                fila.append(f"{media_final:.2f} ± {desv_final:.2f} (n={n_por_replica}x3)")

        filas.append(fila)

    lineas_md = ["| " + " | ".join(fila) + " |" for fila in filas]
    tabla = "\n".join(lineas_md)

    resumen = [
        "",
        "**Metodo de combinacion**: cada celda es la media +- desviacion estandar de las "
        "3 medias replicadas (una por cada ejecucion completa de la rejilla H x B), no de "
        "las instancias individuales combinadas. `(n=20x3)` significa: 20 instancias por "
        "replica, promediadas sobre 3 replicas independientes.",
        f"Ficheros de origen: {', '.join(FICHEROS_REPLICAS)}",
    ]
    if filas_omitidas:
        resumen.append("\n**Filas omitidas** (no combinables todavia):")
        for linea in filas_omitidas:
            resumen.append(f"- {linea}")

    return tabla + "\n" + "\n".join(resumen)


if __name__ == "__main__":
    tabla_md = combinar()
    with open(TABLA_FINAL_FILE, "w", encoding="utf-8") as fh:
        fh.write(tabla_md + "\n")
    print(f"\nTabla final combinada guardada en {TABLA_FINAL_FILE}\n")
    print(tabla_md)
