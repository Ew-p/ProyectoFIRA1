#!/usr/bin/env python3
"""
Separa las filas de un Excel/CSV según la columna C (Sentinel-2 / Landsat)
y genera dos archivos Excel. La primera fila se toma como encabezado.

Uso interactivo (menú en terminal):
    python separar_satelites.py

Uso directo (sin menú):
    python separar_satelites.py datos.xlsx [-o carpeta] [-s hoja]

Requisitos:
    pip install pandas openpyxl      (para .xls antiguo: pip install xlrd)
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

COLUMNA_C = 2  # índice base 0: A=0, B=1, C=2
EXTENSIONES = (".xlsx", ".xlsm", ".xls", ".csv", ".txt")


# ---------------------------------------------------------------- lógica ---
def leer_archivo(ruta: Path, hoja=0) -> pd.DataFrame:
    ext = ruta.suffix.lower()
    if ext in (".xlsx", ".xlsm", ".xls"):
        return pd.read_excel(ruta, sheet_name=hoja, header=0)
    if ext in (".csv", ".txt"):
        for enc in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
            try:
                return pd.read_csv(ruta, sep=None, engine="python",
                                   encoding=enc, header=0)
            except UnicodeDecodeError:
                continue
        raise ValueError("No se pudo decodificar el CSV.")
    raise ValueError(f"Formato no soportado: {ext}")


def separar(df: pd.DataFrame):
    if df.shape[1] <= COLUMNA_C:
        raise ValueError(
            f"El archivo solo tiene {df.shape[1]} columnas; se necesita la C.")
    col = df.iloc[:, COLUMNA_C].astype(str).str.strip()
    es_s = col.str.contains(r"sentinel[\s\-_]*2", case=False, regex=True)
    es_l = col.str.contains(r"landsat", case=False, regex=True)
    return df[es_s], df[es_l], df[~(es_s | es_l)]


def procesar(ruta: Path, carpeta: Path, hoja=0):
    df = leer_archivo(ruta, hoja)
    sentinel, landsat, otros = separar(df)
    carpeta.mkdir(parents=True, exist_ok=True)
    out_s = carpeta / f"{ruta.stem}_Sentinel-2.xlsx"
    out_l = carpeta / f"{ruta.stem}_Landsat.xlsx"
    sentinel.to_excel(out_s, index=False)
    landsat.to_excel(out_l, index=False)
    return len(df), len(sentinel), len(landsat), len(otros), out_s, out_l


# ------------------------------------------------------------- interfaz ----
class C:  # colores ANSI (se desactivan si la terminal no los soporta)
    on = sys.stdout.isatty()
    R = "\033[0m" if on else ""
    B = "\033[1m" if on else ""
    G = "\033[92m" if on else ""
    Y = "\033[93m" if on else ""
    RED = "\033[91m" if on else ""
    CY = "\033[96m" if on else ""
    D = "\033[2m" if on else ""


def limpiar():
    os.system("cls" if os.name == "nt" else "clear")


def banner():
    print(f"{C.CY}{C.B}╔══════════════════════════════════════════╗")
    print("║   SEPARADOR SENTINEL-2 / LANDSAT (col C) ║")
    print(f"╚══════════════════════════════════════════╝{C.R}")


def pausa():
    input(f"\n{C.D}Enter para continuar...{C.R}")


def limpiar_ruta(txt: str) -> Path:
    # Quita comillas (al arrastrar un archivo a la terminal) y expande ~
    return Path(txt.strip().strip('"').strip("'")).expanduser()


def elegir_archivo():
    archivos = sorted(p for p in Path.cwd().iterdir()
                      if p.suffix.lower() in EXTENSIONES and p.is_file())
    print(f"\n{C.B}Archivos en {Path.cwd()}:{C.R}")
    if archivos:
        for i, p in enumerate(archivos, 1):
            print(f"  {C.Y}{i}{C.R}) {p.name}")
    else:
        print(f"  {C.D}(no hay Excel/CSV en esta carpeta){C.R}")
    print(f"\nEscribe el {C.B}número{C.R}, o pega/arrastra una {C.B}ruta{C.R} "
          f"(vacío = cancelar)")
    resp = input(f"{C.CY}> {C.R}").strip()
    if not resp:
        return None
    if resp.isdigit() and 1 <= int(resp) <= len(archivos):
        return archivos[int(resp) - 1]
    ruta = limpiar_ruta(resp)
    if ruta.is_file():
        return ruta
    print(f"{C.RED}No se encontró ese archivo.{C.R}")
    return None


def elegir_hoja(ruta: Path):
    if ruta.suffix.lower() not in (".xlsx", ".xlsm", ".xls"):
        return 0
    try:
        hojas = pd.ExcelFile(ruta).sheet_names
    except Exception as e:
        print(f"{C.RED}No se pudo abrir el Excel: {e}{C.R}")
        return 0
    if len(hojas) == 1:
        return hojas[0]
    print(f"\n{C.B}Hojas disponibles:{C.R}")
    for i, h in enumerate(hojas, 1):
        print(f"  {C.Y}{i}{C.R}) {h}")
    resp = input(f"{C.CY}Hoja [1]: {C.R}").strip()
    if resp.isdigit() and 1 <= int(resp) <= len(hojas):
        return hojas[int(resp) - 1]
    return hojas[0]


def vista_previa(ruta: Path, hoja):
    df = leer_archivo(ruta, hoja)
    print(f"\n{C.B}Filas de datos:{C.R} {len(df)}   "
          f"{C.B}Columnas:{C.R} {df.shape[1]}")
    if df.shape[1] <= COLUMNA_C:
        print(f"{C.RED}El archivo no tiene columna C.{C.R}")
        return
    s, l, o = separar(df)
    print(f"  {C.G}Sentinel-2:{C.R} {len(s)}")
    print(f"  {C.G}Landsat   :{C.R} {len(l)}")
    print(f"  {C.Y}Sin coincidencia:{C.R} {len(o)}")
    print(f"\n{C.B}Primeras filas:{C.R}")
    print(df.head(5).to_string(max_colwidth=25))


def menu():
    ruta = None
    hoja = 0
    salida = None  # None = misma carpeta que el archivo

    while True:
        limpiar()
        banner()
        print(f"\n  Archivo : {C.G + ruta.name + C.R if ruta else C.D + '(ninguno)' + C.R}")
        if ruta and ruta.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
            print(f"  Hoja    : {hoja}")
        print(f"  Salida  : {salida if salida else C.D + '(misma carpeta del archivo)' + C.R}")
        print(f"\n  {C.Y}1{C.R}) Seleccionar archivo")
        print(f"  {C.Y}2{C.R}) Vista previa")
        print(f"  {C.Y}3{C.R}) Cambiar carpeta de salida")
        print(f"  {C.Y}4{C.R}) {C.B}Procesar y separar{C.R}")
        print(f"  {C.Y}0{C.R}) Salir")
        op = input(f"\n{C.CY}Opción > {C.R}").strip()

        try:
            if op == "1":
                nuevo = elegir_archivo()
                if nuevo:
                    ruta = nuevo
                    hoja = elegir_hoja(ruta)
                else:
                    pausa()
            elif op == "2":
                if not ruta:
                    print(f"{C.RED}Primero selecciona un archivo.{C.R}")
                else:
                    vista_previa(ruta, hoja)
                pausa()
            elif op == "3":
                txt = input("Carpeta de salida (vacío = misma del archivo): ").strip()
                salida = limpiar_ruta(txt) if txt else None
            elif op == "4":
                if not ruta:
                    print(f"{C.RED}Primero selecciona un archivo.{C.R}")
                else:
                    carpeta = salida if salida else ruta.parent
                    total, ns, nl, no, os_, ol = procesar(ruta, carpeta, hoja)
                    print(f"\n{C.G}{C.B}✔ Listo{C.R}")
                    print(f"  Analizadas : {total} filas")
                    print(f"  Sentinel-2 : {ns} -> {os_}")
                    print(f"  Landsat    : {nl} -> {ol}")
                    if no:
                        print(f"  {C.Y}Sin coincidencia: {no} filas (no guardadas){C.R}")
                pausa()
            elif op == "0":
                print("¡Hasta luego!")
                return
        except Exception as e:
            print(f"\n{C.RED}Error: {e}{C.R}")
            pausa()


# ----------------------------------------------------------------- main ----
def main():
    parser = argparse.ArgumentParser(
        description="Separa filas por Sentinel-2 / Landsat (columna C). "
                    "Sin argumentos abre el menú interactivo.")
    parser.add_argument("archivo", nargs="?", help="Excel (.xlsx/.xls) o CSV")
    parser.add_argument("-o", "--salida", help="Carpeta de salida")
    parser.add_argument("-s", "--hoja", default=0, help="Hoja del Excel")
    args = parser.parse_args()

    if not args.archivo:
        try:
            menu()
        except (KeyboardInterrupt, EOFError):
            print("\nSaliendo...")
        return

    ruta = limpiar_ruta(args.archivo)
    if not ruta.exists():
        sys.exit(f"Error: no existe '{ruta}'")
    hoja = int(args.hoja) if str(args.hoja).isdigit() else args.hoja
    carpeta = Path(args.salida) if args.salida else ruta.parent
    total, ns, nl, no, os_, ol = procesar(ruta, carpeta, hoja)
    print(f"Analizadas: {total} | Sentinel-2: {ns} -> {os_} | Landsat: {nl} -> {ol}")
    if no:
        print(f"Sin coincidencia: {no} filas (no guardadas)")


if __name__ == "__main__":
    main()