"""
separar_datos.py
----------------
Lee un archivo grande en Excel (.xlsx/.xls/.xlsm) o CSV (fila 1 = características,
columna A = ID tipo AGC_001...AGC_197, un ID puede tener muchas filas con distinta fecha),
ordena por ID y, dentro de cada ID, de la fecha más antigua a la más reciente.
Cada fila se mueve completa, así que sus datos asociados siempre la acompañan.
Genera dos archivos:

    - datos_prediccion     -> solo los IDs de la lista PREDICCION_IDS
    - datos_entrenamiento  -> todo lo demás

Por defecto los resultados salen en el mismo formato que el archivo de entrada
(CSV -> CSV, Excel -> Excel). Se puede forzar con --formato xlsx  o  --formato csv.

Uso:
    python separar_datos.py                        (te pide la ruta del archivo)
    python separar_datos.py mis_datos.csv          (también acepta la ruta directa)
    python separar_datos.py mis_datos.xlsx --hoja "Hoja1" --salida resultados
    python separar_datos.py mis_datos.csv --formato xlsx
    python separar_datos.py mis_datos.xlsx --col-fecha "Fecha"   (si la fecha no está en la columna B)

Requiere:  pip install pandas openpyxl      (para .xls antiguo además: pip install xlrd)
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import pandas as pd

# ----------------------------------------------------------------------------
# IDs que van al archivo de PREDICCIÓN (todo lo demás es ENTRENAMIENTO)
# ----------------------------------------------------------------------------
PREDICCION_IDS = [
    "AGC_003", "AGC_011", "AGC_016", "AGC_025", "AGC_027", "AGC_028", "AGC_029",
    "AGC_030", "AGC_031", "AGC_036", "AGC_039", "AGC_042", "AGC_043", "AGC_050",
    "AGC_052", "AGC_061", "AGC_063", "AGC_065", "AGC_071", "AGC_075", "AGC_076",
    "AGC_082", "AGC_085", "AGC_086", "AGC_087", "AGC_088", "AGC_091", "AGC_093",
    "AGC_094", "AGC_095", "AGC_107", "AGC_113", "AGC_116", "AGC_118", "AGC_120",
    "AGC_126", "AGC_128", "AGC_129", "AGC_130", "AGC_136", "AGC_142", "AGC_143",
    "AGC_147", "AGC_149", "AGC_153", "AGC_154", "AGC_156", "AGC_158", "AGC_173",
    "AGC_174", "AGC_175", "AGC_177", "AGC_182", "AGC_183", "AGC_186", "AGC_188",
    "AGC_190", "AGC_191", "AGC_192",
]

PATRON_ID = re.compile(r"^\s*AGC[\s_\-]*0*(\d+)\s*$", re.IGNORECASE)
EXT_EXCEL = {".xlsx", ".xlsm", ".xls"}
EXT_CSV = {".csv"}


def normalizar_id(valor):
    """Convierte variantes ('agc_3', 'AGC-003 ', 'AGC 003') a 'AGC_003'.
    Regresa None si no tiene el formato esperado."""
    m = PATRON_ID.match(str(valor))
    if not m:
        return None
    return f"AGC_{int(m.group(1)):03d}"


FORMATOS_FECHA = ["%d/%m/%Y", "%d/%m/%Y %H:%M:%S", "%d-%m-%Y", "%d.%m.%Y",
                  "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]


def parsear_fechas(serie):
    """Convierte una columna a fechas reales (día/mes/año). Lo que no se pueda leer queda NaT.
    Solo se usa para ORDENAR: la columna original se guarda tal cual estaba."""
    if pd.api.types.is_datetime64_any_dtype(serie):
        return serie
    texto = serie.astype(str).str.strip()
    resultado = pd.Series(pd.NaT, index=serie.index, dtype="datetime64[ns]")
    for fmt in FORMATOS_FECHA:
        pendiente = resultado.isna() & serie.notna()
        if not pendiente.any():
            break
        resultado[pendiente] = pd.to_datetime(texto[pendiente], format=fmt, errors="coerce")
    return resultado


def detectar_col_fecha(df, indicada=None):
    """Usa la columna indicada; si no, la columna B; si no parece de fechas,
    busca una columna con 'fecha' o 'date' en el nombre."""
    if indicada is not None:
        if indicada not in df.columns:
            sys.exit(f"[ERROR] No existe la columna de fecha '{indicada}'. "
                     f"Columnas disponibles: {list(df.columns)[:10]}...")
        return indicada
    candidatas = list(df.columns[1:2]) + [
        c for c in df.columns[2:] if re.search(r"fecha|date", str(c), re.IGNORECASE)
    ]
    for c in candidatas:
        con_dato = df[c].notna().sum()
        if con_dato and parsear_fechas(df[c]).notna().sum() / con_dato >= 0.8:
            return c
    sys.exit("[ERROR] No pude identificar la columna de fecha. "
             "Indícala con --col-fecha \"NombreDeLaColumna\"")


def pedir_ruta(ruta=None):
    """Pide la ruta del archivo por consola hasta que sea un Excel o CSV válido.
    Acepta rutas con comillas (por ejemplo, al arrastrar el archivo a la terminal)."""
    while True:
        if not ruta:
            ruta = input("Ruta del archivo Excel o CSV (puedes arrastrarlo aquí): ")
        ruta = ruta.strip().strip('"').strip("'")
        p = Path(ruta).expanduser()
        if not p.is_file():
            print(f"[ERROR] No encontré el archivo: {ruta}")
        elif p.suffix.lower() not in EXT_EXCEL | EXT_CSV:
            print(f"[ERROR] Formato no soportado ({p.suffix}). Usa .xlsx, .xls, .xlsm o .csv")
        else:
            return p
        ruta = None


def leer_datos(ruta, hoja=0):
    """Lee Excel o CSV. Regresa (DataFrame, formato, separador_csv).
    En CSV detecta solo la codificación y el separador (coma, punto y coma, tab, |)."""
    ext = ruta.suffix.lower()

    if ext in EXT_EXCEL:
        return pd.read_excel(ruta, sheet_name=hoja), "xlsx", ","

    # --- CSV ---
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            with open(ruta, "r", encoding=enc, newline="") as f:
                muestra = f.read(20000)
            break
        except UnicodeDecodeError:
            continue
    try:
        sep = csv.Sniffer().sniff(muestra, delimiters=",;\t|").delimiter
    except csv.Error:
        sep = ","
    print(f"CSV detectado -> codificación: {enc}, separador: {sep!r}")
    return pd.read_csv(ruta, sep=sep, encoding=enc), "csv", sep


def guardar_datos(df, carpeta, nombre, formato, sep=","):
    """Guarda el DataFrame como xlsx (con encabezado congelado) o csv.
    Regresa la ruta del archivo creado."""
    if formato == "csv":
        ruta = carpeta / f"{nombre}.csv"
        # utf-8-sig para que Excel abra bien los acentos
        df.to_csv(ruta, index=False, sep=sep, encoding="utf-8-sig")
    else:
        ruta = carpeta / f"{nombre}.xlsx"
        with pd.ExcelWriter(ruta, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Datos")
            ws = writer.sheets["Datos"]
            # Columnas con fechas reales: mostrarlas como día/mes/año
            for i, col in enumerate(df.columns, start=1):
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    solo_fecha = (df[col].dropna() == df[col].dropna().dt.normalize()).all()
                    formato = "DD/MM/YYYY" if solo_fecha else "DD/MM/YYYY HH:MM:SS"
                    for fila in ws.iter_rows(min_row=2, min_col=i, max_col=i):
                        fila[0].number_format = formato
                    ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = 14
            ws.freeze_panes = "B2"  # deja fija la fila 1 y la columna de IDs
            ws.column_dimensions["A"].width = 12
    return ruta


def main():
    parser = argparse.ArgumentParser(description="Separa datos en entrenamiento y predicción.")
    parser.add_argument("archivo", nargs="?", default=None,
                        help="Ruta del archivo (.xlsx, .xls, .xlsm o .csv). Si se omite, el programa la pide.")
    parser.add_argument("--hoja", default=0,
                        help="Solo Excel: nombre o índice de la hoja (por defecto la primera)")
    parser.add_argument("--salida", default=".", help="Carpeta donde se guardan los resultados")
    parser.add_argument("--col-fecha", default=None,
                        help="Nombre de la columna de fecha (por defecto se usa la columna B)")
    parser.add_argument("--formato", choices=["xlsx", "csv"], default=None,
                        help="Formato de salida (por defecto el mismo que el de entrada)")
    args = parser.parse_args()

    ruta_entrada = pedir_ruta(args.archivo)
    hoja = int(args.hoja) if str(args.hoja).isdigit() else args.hoja
    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)

    # 1) Leer el archivo -------------------------------------------------------
    df, formato_in, sep = leer_datos(ruta_entrada, hoja)
    formato_out = args.formato or formato_in
    col_id = df.columns[0]
    print(f"Archivo leído: {df.shape[0]} filas x {df.shape[1]} columnas")
    print(f"Columna de ID: '{col_id}'")

    # 2) Normalizar IDs y detectar filas raras ---------------------------------
    df["_id_norm"] = df[col_id].apply(normalizar_id)

    invalidas = df[df["_id_norm"].isna()]
    if not invalidas.empty:
        print(f"\n[AVISO] {len(invalidas)} fila(s) con ID que no sigue el formato AGC_###.")
        print("        Se guardan en 'filas_revisar' y NO entran a ningún set.")
        guardar_datos(invalidas.drop(columns="_id_norm"), salida, "filas_revisar", formato_out, sep)
    df = df.dropna(subset=["_id_norm"]).copy()

    # 3) Fechas ----------------------------------------------------------------
    col_fecha = detectar_col_fecha(df, args.col_fecha)
    df["_fecha"] = parsear_fechas(df[col_fecha])
    print(f"Columna de fecha: '{col_fecha}'")

    sin_fecha = df["_fecha"].isna().sum()
    if sin_fecha:
        print(f"\n[AVISO] {sin_fecha} fila(s) con fecha vacía o ilegible; "
              "quedan al final de su ID.")

    misma = df.duplicated(subset=["_id_norm", "_fecha"], keep=False) & df["_fecha"].notna()
    if misma.any():
        print(f"[AVISO] {misma.sum()} fila(s) comparten ID y fecha exactas con otra fila "
              "(se respeta el orden original entre ellas).")

    # 4) Ordenar: primero por ID, luego por fecha (antigua -> reciente) --------
    df[col_id] = df["_id_norm"]  # deja los IDs limpios y uniformes
    df["_num"] = df["_id_norm"].str[4:].astype(int)
    df = df.sort_values(["_num", "_fecha"], kind="stable", na_position="last").reset_index(drop=True)

    # 5) Separar ---------------------------------------------------------------
    ids_pred = {normalizar_id(i) for i in PREDICCION_IDS}
    es_pred = df["_id_norm"].isin(ids_pred)

    limpiar = ["_id_norm", "_num", "_fecha"]
    df_pred = df[es_pred].drop(columns=limpiar).reset_index(drop=True)
    df_train = df[~es_pred].drop(columns=limpiar).reset_index(drop=True)

    faltantes = sorted(ids_pred - set(df["_id_norm"]))
    if faltantes:
        print(f"\n[AVISO] IDs de predicción que NO están en el archivo ({len(faltantes)}): "
              f"{', '.join(faltantes)}")

    # 6) Guardar ---------------------------------------------------------------
    ruta_train = guardar_datos(df_train, salida, "datos_entrenamiento", formato_out, sep)
    ruta_pred = guardar_datos(df_pred, salida, "datos_prediccion", formato_out, sep)

    print("\n=== Resultado ===")
    print(f"Entrenamiento: {len(df_train)} filas, {df_train[col_id].nunique()} IDs -> {ruta_train}")
    print(f"Predicción:    {len(df_pred)} filas, {df_pred[col_id].nunique()} IDs -> {ruta_pred}")
    print(f"Total:         {len(df_train) + len(df_pred)} (de {len(df)} filas válidas)")


if __name__ == "__main__":
    sys.exit(main())