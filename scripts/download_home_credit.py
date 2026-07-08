"""
Descarga UNICAMENTE application_train.csv de la competicion Home Credit
Default Risk y lo copia a la ruta que espera load_home_credit().

Requisito previo obligatorio (independiente de este script): haber
aceptado las reglas de la competicion en
https://www.kaggle.com/competitions/home-credit-default-risk/rules
-- sin esto, kagglehub devuelve un error 403 sin importar las
credenciales usadas.

Uso: python scripts/download_home_credit.py
"""
import shutil
from pathlib import Path

import kagglehub

DESTINO = Path("data/home-credit-default/application_train.csv")


def main():
    print("Descargando application_train.csv (puede tardar por su tamano, ~158 MB)...")
    ruta_descarga = kagglehub.competition_download(
        "home-credit-default-risk",
        path="application_train.csv",
    )
    ruta_descarga = Path(ruta_descarga)

    # kagglehub puede devolver la ruta al fichero directamente o a la
    # carpeta que lo contiene, segun version -- se cubren ambos casos.
    if ruta_descarga.is_dir():
        ruta_descarga = ruta_descarga / "application_train.csv"

    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ruta_descarga, DESTINO)

    print(f"Copiado a: {DESTINO.resolve()}")
    print(f"Tamano: {DESTINO.stat().st_size / (1024*1024):.1f} MB")


if __name__ == "__main__":
    main()
