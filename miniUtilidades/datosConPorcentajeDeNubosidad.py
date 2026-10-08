"""
Analiza el efecto de filtrar registros satelitales por porcentaje de nubosidad.

Sirve para CUALQUIERA de tus dos CSV (Sentinel-2 o Landsat), sin importar
cuántas columnas de características tenga cada uno: las detecta solas.

USO
---
python3 analizar_nubosidad.py archivo.csv --umbral 20
python3 analizar_nubosidad.py archivo.csv --umbrales 5,10,20,30,50,70

Columnas que espera encontrar (nombres flexibles, no exactos):
- Un ID de parcela      -> cualquier columna que contenga "ID" o "POLIGONO"
- Una fecha              -> cualquier columna que contenga "fecha"
- Un % de nubosidad      -> cualquier columna que contenga "nub"
- Todo lo demás numérico se trata como "característica" a evaluar.

SALIDAS (se guardan junto al CSV de entrada, con sufijo _reporte)
- <archivo>_resumen_parcela.csv   -> cuánto pierde cada parcela
- <archivo>_resumen_mes_anio.csv  -> cuánto se pierde por mes y año
- <archivo>_huecos.csv            -> combinaciones parcela-año-mes que
                                      quedarían SIN ningún dato tras filtrar
"""
import argparse
import glob
import os
import sys
import numpy as np
import pandas as pd

pd.set_option("display.width", 160)
pd.set_option("display.max_rows", 60)


# --------------------------------------------------------------------------
# 1) Detección flexible de columnas
REPORT_SUFFIXES = ("_resumen_parcela.csv", "_resumen_mes_anio.csv", "_huecos.csv")


def expandir_rutas(rutas):
    """Convierte una lista de rutas (archivos y/o carpetas) en una lista plana
    de archivos .csv/.xlsx/.xls, ignorando los reportes que el propio script
    ya generó antes (para no reprocesarlos si vuelves a apuntar a la carpeta)."""
    encontrados = []
    for r in rutas:
        r = r.strip().strip('"').strip("'")
        if os.path.isdir(r):
            for ext in ("*.csv", "*.xlsx", "*.xls"):
                encontrados.extend(sorted(glob.glob(os.path.join(r, ext))))
        else:
            encontrados.append(r)
    vistos, limpio = set(), []
    for a in encontrados:
        if a.endswith(REPORT_SUFFIXES):
            continue
        if a in vistos:
            continue
        vistos.add(a)
        limpio.append(a)
    return limpio


# --------------------------------------------------------------------------
def detectar_columnas(df: pd.DataFrame):
    cols = df.columns
    col_id = next((c for c in cols if "poligon" in c.lower() or c.lower() in ("id", "id_poligono")), None)
    col_fecha = next((c for c in cols if "fecha" in c.lower()), None)
    col_nub = next((c for c in cols if "nub" in c.lower()), None)
    col_sensor = next((c for c in cols if "sensor" in c.lower()), None)

    if col_id is None or col_fecha is None or col_nub is None:
        faltan = [n for n, c in [("ID de parcela", col_id), ("fecha", col_fecha), ("nubosidad", col_nub)] if c is None]
        sys.exit(f"No encontré columna(s) de: {', '.join(faltan)}. Revisa los nombres del CSV.")

    meta = {col_id, col_fecha, col_nub}
    if col_sensor:
        meta.add(col_sensor)
    # características = todo lo numérico que no sea metadato
    caracteristicas = [c for c in cols if c not in meta and pd.api.types.is_numeric_dtype(df[c])]
    return col_id, col_fecha, col_nub, col_sensor, caracteristicas


def leer_tabla(path: str) -> pd.DataFrame:
    """Lee .csv, .xlsx o .xls. Para CSV, prueba varias codificaciones típicas
    de Windows/Excel en español antes de rendirse."""
    ext = path.rsplit(".", 1)[-1].lower()
    if ext in ("xlsx", "xls"):
        try:
            return pd.read_excel(path)
        except ImportError:
            sys.exit("Para leer .xlsx necesitas la librería 'openpyxl'. "
                      "Instálala con:  pip install openpyxl")
    # CSV: probar codificaciones en orden
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
        except FileNotFoundError:
            raise
    # último intento: deja que pandas reporte el error real
    return pd.read_csv(path)


def cargar(path: str):
    df = leer_tabla(path)
    col_id, col_fecha, col_nub, col_sensor, feats = detectar_columnas(df)

    # fecha: intenta dd/mm/aaaa y si no, deja que pandas infiera
    fecha = pd.to_datetime(df[col_fecha], format="%d/%m/%Y", errors="coerce")
    if fecha.isna().mean() > 0.5:
        fecha = pd.to_datetime(df[col_fecha], errors="coerce")
    df["_fecha"] = fecha
    df["_anio"] = df["_fecha"].dt.year
    df["_mes"] = df["_fecha"].dt.month
    df["_id"] = df[col_id]
    df["_nub"] = pd.to_numeric(df[col_nub], errors="coerce")
    df["_tiene_dato"] = df[feats].notna().any(axis=1)
    return df, feats, col_sensor


# --------------------------------------------------------------------------
# 2) Inventario previo (antes de filtrar nada)
# --------------------------------------------------------------------------
def inventario_previo(df, feats, col_sensor):
    print("=" * 70)
    print("INVENTARIO PREVIO (datos tal cual llegaron)")
    print("=" * 70)
    print(f"Filas totales:            {len(df):,}")
    print(f"Parcelas (ID) distintas:  {df['_id'].nunique():,}")
    print(f"Características numéricas detectadas ({len(feats)}): {feats}")
    if col_sensor:
        print("Sensores presentes:", df[col_sensor].value_counts().to_dict())
    print(f"Rango de fechas:          {df['_fecha'].min().date()}  a  {df['_fecha'].max().date()}")
    print(f"Fechas no interpretables: {df['_fecha'].isna().sum()} filas")
    print(f"Filas sin NINGÚN dato en las características: {(~df['_tiene_dato']).sum():,} "
          f"({(~df['_tiene_dato']).mean()*100:.1f}%)")
    print("\n% de valores presentes por característica:")
    print((df[feats].notna().mean() * 100).round(1).sort_values(ascending=False).to_string())
    print("\nDistribución de nubosidad (%):")
    print(df["_nub"].describe().round(2).to_string())
    print("\nObservaciones válidas (con algún dato) por año:")
    print(df[df["_tiene_dato"]].groupby("_anio").size().to_string())
    print("\nObservaciones válidas (con algún dato) por mes:")
    print(df[df["_tiene_dato"]].groupby("_mes").size().to_string())
    print()


# --------------------------------------------------------------------------
# 3) Sesgo: compara cada registro contra el valor "sin nubes" de su
#    misma parcela + mismo mes (línea base local), para no confundir
#    el sesgo por nube con el ciclo estacional normal del cultivo.
# --------------------------------------------------------------------------
def calcular_sesgo(df, feats):
    d = df[df["_tiene_dato"]].copy()
    d["_id_mes"] = d["_id"].astype(str) + "_" + d["_mes"].astype(str)
    difs = {}
    for f in feats:
        base = d[d["_nub"] == 0].groupby("_id_mes")[f].mean()
        difs["_dif_" + f] = d[f] - d["_id_mes"].map(base)
    return pd.concat([d, pd.DataFrame(difs, index=d.index)], axis=1)


def resumen_sesgo_por_umbral(d, feats, umbral):
    """Sesgo promedio (en las filas que SE ELIMINARÍAN) frente a su línea base."""
    elim = d[d["_nub"] > umbral]
    filas = []
    for f in feats:
        dif = elim["_dif_" + f].dropna()
        if len(dif) == 0:
            continue
        sd = d[f].std()
        filas.append({
            "caracteristica": f,
            "n_comparable": len(dif),
            "sesgo_promedio": dif.mean(),
            "sesgo_en_desv_std": dif.mean() / sd if sd else np.nan,
        })
    return pd.DataFrame(filas).round(4)


# --------------------------------------------------------------------------
# 4) Impacto de aplicar el umbral: cuántos datos se eliminan y dónde duele
# --------------------------------------------------------------------------
def impacto_umbral(df, feats, umbral, prefijo_salida):
    total = df["_tiene_dato"].sum()
    elim_mask = df["_tiene_dato"] & (df["_nub"] > umbral)
    n_elim = elim_mask.sum()

    print("-" * 70)
    print(f"UMBRAL: se eliminan registros con nubosidad > {umbral}%")
    print("-" * 70)
    print(f"Filas con dato antes de filtrar: {total:,}")
    print(f"Filas que se eliminarían:        {n_elim:,}  ({n_elim/total*100:.1f}% del total con dato)")
    print(f"Filas que quedarían:              {total - n_elim:,}")

    # --- sesgo de lo que se elimina ---
    d = calcular_sesgo(df, feats)
    tabla_sesgo = resumen_sesgo_por_umbral(d, feats, umbral)
    if not tabla_sesgo.empty:
        print("\nSesgo promedio de las filas eliminadas (frente a su propia parcela+mes sin nubes):")
        print(tabla_sesgo.sort_values("sesgo_en_desv_std", key=abs, ascending=False).to_string(index=False))
        peor = tabla_sesgo.loc[tabla_sesgo.sesgo_en_desv_std.abs().idxmax()]
        print(f"\n  -> El sesgo más fuerte es en '{peor.caracteristica}': "
              f"{peor.sesgo_promedio:+.4f} ({peor.sesgo_en_desv_std:+.2f} desviaciones estándar).")

    # --- impacto por parcela ---
    por_id_total = df[df["_tiene_dato"]].groupby("_id").size()
    por_id_elim = df[elim_mask].groupby("_id").size().reindex(por_id_total.index, fill_value=0)
    resumen_id = pd.DataFrame({
        "obs_totales": por_id_total,
        "obs_eliminadas": por_id_elim,
        "pct_eliminado": (por_id_elim / por_id_total * 100).round(1),
    }).sort_values("pct_eliminado", ascending=False)
    resumen_id.to_csv(prefijo_salida + "_resumen_parcela.csv")
    print(f"\nParcelas que perderían más del 40% de sus observaciones: "
          f"{(resumen_id.pct_eliminado > 40).sum()} de {len(resumen_id)}")
    print("Top 5 parcelas más afectadas:")
    print(resumen_id.head(5).to_string())

    # --- impacto por mes y año ---
    por_my_total = df[df["_tiene_dato"]].groupby(["_anio", "_mes"]).size()
    por_my_elim = df[elim_mask].groupby(["_anio", "_mes"]).size().reindex(por_my_total.index, fill_value=0)
    resumen_my = pd.DataFrame({
        "obs_totales": por_my_total,
        "obs_eliminadas": por_my_elim,
        "pct_eliminado": (por_my_elim / por_my_total * 100).round(1),
    }).sort_values("pct_eliminado", ascending=False)
    resumen_my.to_csv(prefijo_salida + "_resumen_mes_anio.csv")
    print("\nMeses/años más afectados (mayor % eliminado):")
    print(resumen_my.head(8).to_string())

    # --- huecos: parcela-año-mes que se quedarían en CERO datos ---
    full = df[df["_tiene_dato"]].groupby(["_id", "_anio", "_mes"]).size()
    quedan = df[df["_tiene_dato"] & (df["_nub"] <= umbral)].groupby(["_id", "_anio", "_mes"]).size()
    quedan = quedan.reindex(full.index, fill_value=0)
    huecos = full[quedan == 0]
    if len(huecos):
        huecos.rename("obs_perdidas").reset_index().to_csv(prefijo_salida + "_huecos.csv", index=False)
    print(f"\nCombinaciones parcela-año-mes que se quedarían SIN NINGÚN dato: "
          f"{len(huecos)} de {len(full)} ({len(huecos)/len(full)*100:.1f}%)")
    if len(huecos):
        print("(detalle guardado en", prefijo_salida + "_huecos.csv)")
        print("Se concentran en estos meses:",
              huecos.reset_index()["_mes"].value_counts().sort_index().to_dict())
    print()
    return resumen_id, resumen_my, huecos


def comparar_umbrales(df, feats, umbrales):
    print("=" * 70)
    print("COMPARACIÓN RÁPIDA ENTRE VARIOS UMBRALES")
    print("=" * 70)
    resumen = []
    total = df["_tiene_dato"].sum()
    d = calcular_sesgo(df, feats)
    for u in umbrales:
        n_elim = (df["_tiene_dato"] & (df["_nub"] > u)).sum()
        tabla = resumen_sesgo_por_umbral(d, feats, u)
        peor = tabla.sesgo_en_desv_std.abs().max() if not tabla.empty else np.nan
        resumen.append({"umbral": u, "filas_eliminadas": n_elim,
                         "pct_eliminado": round(n_elim / total * 100, 1),
                         "peor_sesgo_en_sd": round(peor, 3) if pd.notna(peor) else np.nan})
    print(pd.DataFrame(resumen).to_string(index=False))
    print()


def pausa():
    input("\n(Enter para continuar...) ")


def pedir_ruta():
    while True:
        entrada = input("Ruta de un CSV/XLSX, o de una CARPETA con varios: ").strip().strip('"').strip("'")
        if os.path.isdir(entrada):
            archivos = expandir_rutas([entrada])
            if not archivos:
                print(f"  No encontré .csv/.xlsx en '{entrada}'.\n")
                continue
            print(f"  Encontré {len(archivos)} archivo(s) en esa carpeta:")
            for a in archivos:
                print("   -", os.path.basename(a))
            u_txt = input("  Umbral a aplicar a todos [Enter = 20]: ").strip()
            try:
                u = float(u_txt) if u_txt else 20.0
            except ValueError:
                print("  Valor no válido, usando 20."); u = 20.0
            procesar_lote(archivos, u)
            print("  Listo el lote de esa carpeta. Puedes darme otra ruta, o un archivo para seguir en el menú normal.\n")
            continue
        try:
            df, feats, col_sensor = cargar(entrada)
            return entrada, df, feats, col_sensor
        except FileNotFoundError:
            print(f"  No encontré el archivo '{entrada}'. Intenta de nuevo.\n")
        except SystemExit as e:
            print(f"  {e}\n")


def pedir_flotantes(msg, default):
    txt = input(f"{msg} [Enter = {default}]: ").strip()
    if not txt:
        txt = default
    try:
        return [float(x) for x in txt.split(",")]
    except ValueError:
        print("  Entrada no válida, usando el valor por defecto.")
        return [float(x) for x in default.split(",")]


def menu_interactivo(ruta_inicial=None):
    print("\n" + "=" * 70)
    print("  ANALIZADOR DE NUBOSIDAD — RETO AGROCEBADA")
    print("=" * 70)

    if ruta_inicial:
        try:
            df, feats, col_sensor = cargar(ruta_inicial)
            ruta = ruta_inicial
        except (FileNotFoundError, SystemExit) as e:
            print(f"  No pude abrir '{ruta_inicial}' ({e}). Pídelo de nuevo.\n")
            ruta, df, feats, col_sensor = pedir_ruta()
    else:
        ruta, df, feats, col_sensor = pedir_ruta()

    inventario_previo(df, feats, col_sensor)
    pausa()

    while True:
        print("\n" + "-" * 70)
        print(f"Archivo actual: {ruta}")
        print("-" * 70)
        print("  1) Comparar varios umbrales de nubosidad (vista rápida)")
        print("  2) Analizar un umbral específico (detalle por parcela/mes/año)")
        print("  3) Ver de nuevo el inventario de este archivo")
        print("  4) Cargar otro archivo CSV")
        print("  5) Procesar VARIOS archivos de un jalón (modo lote)")
        print("  6) Salir")
        op = input("Elige una opción (1-6): ").strip()

        if op == "1":
            umbrales = pedir_flotantes("Umbrales a comparar, separados por coma", "5,10,20,30,50,70")
            comparar_umbrales(df, feats, umbrales)
            pausa()

        elif op == "2":
            txt = input("¿Qué umbral quieres evaluar (%)? [Enter = 20]: ").strip()
            try:
                umbral = float(txt) if txt else 20.0
            except ValueError:
                print("  Valor no válido, usando 20."); umbral = 20.0
            salida = ruta.rsplit(".", 1)[0]
            impacto_umbral(df, feats, umbral, salida)
            pausa()

        elif op == "3":
            inventario_previo(df, feats, col_sensor)
            pausa()

        elif op == "4":
            ruta, df, feats, col_sensor = pedir_ruta()
            inventario_previo(df, feats, col_sensor)
            pausa()

        elif op == "5":
            txt = input("Ruta de una CARPETA, o varias rutas de archivo separadas por coma: ").strip()
            if os.path.isdir(txt.strip().strip('"').strip("'")):
                rutas = expandir_rutas([txt])
            else:
                rutas = [r.strip().strip('"').strip("'") for r in txt.split(",") if r.strip()]
            if len(rutas) < 2:
                print("  Necesito al menos 2 archivos (o una carpeta que contenga varios).")
            else:
                u_txt = input("Umbral a aplicar a todos [Enter = 20]: ").strip()
                try:
                    u = float(u_txt) if u_txt else 20.0
                except ValueError:
                    print("  Valor no válido, usando 20."); u = 20.0
                procesar_lote(rutas, u)
            pausa()

        elif op == "6":
            print("Listo, hasta luego.")
            break

        else:
            print("  Opción no reconocida, elige un número del 1 al 5.")


def procesar_lote(rutas, umbral):
    print("#" * 70)
    print(f"# MODO LOTE: {len(rutas)} archivo(s), umbral fijo = {umbral}%")
    print("#" * 70)
    resumen_general = []
    for ruta in rutas:
        print("\n" + "█" * 70)
        print(f"█  ARCHIVO: {ruta}")
        print("█" * 70)
        try:
            df, feats, col_sensor = cargar(ruta)
        except (FileNotFoundError, SystemExit) as e:
            print(f"  No pude abrir '{ruta}': {e}. Se salta.")
            continue
        inventario_previo(df, feats, col_sensor)
        salida = ruta.rsplit(".", 1)[0]
        resumen_id, resumen_my, huecos = impacto_umbral(df, feats, umbral, salida)
        total = df["_tiene_dato"].sum()
        elim = (df["_tiene_dato"] & (df["_nub"] > umbral)).sum()
        resumen_general.append({
            "archivo": ruta, "filas_con_dato": total, "filas_eliminadas": elim,
            "pct_eliminado": round(elim / total * 100, 1) if total else np.nan,
            "parcelas": df["_id"].nunique(), "huecos_parcela_mes_anio": len(huecos),
        })
    print("\n" + "=" * 70)
    print(f"RESUMEN GENERAL DEL LOTE (umbral = {umbral}%)")
    print("=" * 70)
    print(pd.DataFrame(resumen_general).to_string(index=False))
    print("\nPor cada archivo se guardaron: _resumen_parcela.csv, _resumen_mes_anio.csv"
          " y _huecos.csv (si aplica), junto al archivo original.\n")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("archivo", nargs="*", default=None,
                     help="Uno o varios CSV/XLSX, y/o una CARPETA que los contenga. "
                          "Si el resultado son 2+ archivos, corre en modo LOTE automáticamente.")
    ap.add_argument("--umbral", type=float, default=None,
                     help="Un solo umbral de nubosidad (%%) a evaluar, p.ej. 20. "
                          "En modo lote, si no se indica, se usa 20 por defecto.")
    ap.add_argument("--umbrales", type=str, default=None,
                     help="Modo directo (un solo archivo): lista de umbrales a comparar, separados por coma")
    args = ap.parse_args()

    archivos = expandir_rutas(args.archivo) if args.archivo else []

    # ---- MODO LOTE: 2+ archivos (venidos de una carpeta o listados a mano) ----
    if len(archivos) > 1:
        procesar_lote(archivos, args.umbral if args.umbral is not None else 20.0)
        return
    if args.archivo and not archivos:
        sys.exit(f"No encontré ningún .csv/.xlsx en: {args.archivo}")

    archivo = archivos[0] if archivos else None

    # ---- Modo directo (un archivo + flags) ----
    if args.umbral is not None or args.umbrales is not None:
        if not archivo:
            sys.exit("En modo directo necesitas indicar el archivo: analizar_nubosidad.py archivo.csv --umbral 20")
        df, feats, col_sensor = cargar(archivo)
        inventario_previo(df, feats, col_sensor)
        salida = archivo.rsplit(".", 1)[0]
        if args.umbral is not None:
            impacto_umbral(df, feats, args.umbral, salida)
        else:
            comparar_umbrales(df, feats, [float(x) for x in args.umbrales.split(",")])
        return

    # ---- Modo normal: interfaz de terminal con menú (un solo archivo o ninguno) ----
    try:
        menu_interactivo(archivo)
    except (KeyboardInterrupt, EOFError):
        print("\nInterrumpido por el usuario. Hasta luego.")


if __name__ == "__main__":
    main()