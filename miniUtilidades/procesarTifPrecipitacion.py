"""
Extrae, por parcela, el valor de cada raster mensual (temperatura, precipitación,
o cualquier variable con el mismo formato) y genera UN EXCEL POR AÑO con todos
los meses disponibles. Después decides qué filtrar.

ESTRUCTURA QUE ESPERA (la misma en que ya vienen tus .tif)
    carpeta_raiz/
        2022/  VARIABLE_2022_01.tif  ...  VARIABLE_2022_12.tif
        2023/  ...
        2024/  ...
        2025/  ...
El nombre exacto no importa, mientras termine en _AAAA_MM.tif (p.ej. PREC_2022_01.tif,
Tmin_2023_07.tif, Tmax_2024_11.tif). La georreferencia se lee directo de las
etiquetas GeoTIFF del archivo (no hay coordenadas escritas a mano en el script),
así que funciona igual para precipitación, temperatura, o lo que sea con esa
misma resolución/formato.

USO
---
python3 extraer_tif_por_anio.py --tifs "ruta/Precipitacion_mensual" --shp "ruta/Parcelas_Reto_AGC_CONJUNTO.shp"
python3 extraer_tif_por_anio.py --tifs "ruta/Tmin" --shp "ruta/Parcelas_Reto_AGC_CONJUNTO.shp" --salida ./salidas_tmin

Si no tienes el .shp a la mano pero ya tienes un CSV con ID_POLIGONO, lon, lat
(y opcionalmente Municipio), usa --centroides en vez de --shp:
python3 extraer_tif_por_anio.py --tifs "ruta/Precipitacion_mensual" --centroides mis_centroides.csv

SALIDA
------
Un archivo <variable>_<anio>.xlsx por cada año encontrado, con una fila por
parcela y una columna por mes (01..12), más los metadatos de la parcela.
"""
import argparse
import glob
import os
import re
import struct
import sys

import numpy as np
import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
MESES = [f"{m:02d}" for m in range(1, 13)]


# ==========================================================================
# 1) Shapefile mínimo (sin geopandas): ID, municipio, área, conjunto, centroide
# ==========================================================================
def _read_dbf(path):
    b = open(path, "rb").read()
    n, hl, rl = struct.unpack("<xxxxIHH", b[:12])
    fields = []
    off = 32
    while b[off] != 0x0D:
        nm = b[off:off + 11].split(b"\x00")[0].decode()
        t = chr(b[off + 11])
        ln = b[off + 16]
        fields.append((nm, t, ln))
        off += 32
    rows = []
    for i in range(n):
        r = b[hl + i * rl:hl + (i + 1) * rl]
        pos = 1
        row = {}
        for nm, t, ln in fields:
            v = r[pos:pos + ln].decode("utf-8", "ignore").strip()
            pos += ln
            row[nm] = (float(v) if v not in ("", "*" * ln) else np.nan) if t in "NF" else v
        rows.append(row)
    return pd.DataFrame(rows)


def _read_shp_centroids(path):
    b = open(path, "rb").read()
    pos = 100
    centroids = []
    while pos < len(b):
        _, cl = struct.unpack(">ii", b[pos:pos + 8])
        rec = b[pos + 8:pos + 8 + cl * 2]
        pos += 8 + cl * 2
        t = struct.unpack("<i", rec[:4])[0]
        if t == 0:
            centroids.append((np.nan, np.nan))
            continue
        npart, npts = struct.unpack("<ii", rec[36:44])
        parts = list(struct.unpack(f"<{npart}i", rec[44:44 + 4 * npart]))
        o = 44 + 4 * npart
        pts = np.frombuffer(rec[o:o + 16 * npts], dtype="<f8").reshape(-1, 2)
        parts.append(npts)
        tot = cx = cy = 0.0
        for i in range(npart):
            ring = pts[parts[i]:parts[i + 1]]
            x, y = ring[:, 0], ring[:, 1]
            c = x[:-1] * y[1:] - x[1:] * y[:-1]
            a = c.sum() / 2
            tot += a
            cx += ((x[:-1] + x[1:]) * c).sum() / 6
            cy += ((y[:-1] + y[1:]) * c).sum() / 6
        centroids.append((cx / tot, cy / tot) if tot else (np.nan, np.nan))
    return centroids


def cargar_parcelas(shp_path=None, centroides_csv=None):
    if centroides_csv:
        df = pd.read_csv(centroides_csv)
        cols_lower = {c.lower(): c for c in df.columns}
        id_col = next((cols_lower[c] for c in cols_lower if "polig" in c or c == "id"), None)
        lon_col = next((cols_lower[c] for c in cols_lower if "lon" in c), None)
        lat_col = next((cols_lower[c] for c in cols_lower if "lat" in c), None)
        if not (id_col and lon_col and lat_col):
            sys.exit("El CSV de --centroides necesita columnas de ID, lon y lat (nombres flexibles).")
        df = df.rename(columns={id_col: "ID_POLIGONO", lon_col: "lon", lat_col: "lat"})
        return df

    if shp_path:
        base = shp_path.rsplit(".", 1)[0]
        dbf = _read_dbf(base + ".dbf")
        cen = _read_shp_centroids(base + ".shp")
        dbf["lon"], dbf["lat"] = zip(*cen)
        id_col = next((c for c in dbf.columns if "polig" in c.lower() or c.lower() == "id"), None)
        if id_col:
            dbf = dbf.rename(columns={id_col: "ID_POLIGONO"})
        return dbf

    sys.exit("Necesitas indicar --shp o --centroides para saber dónde está cada parcela.")


# ==========================================================================
# 2) Lectura genérica de un GeoTIFF: valor + georreferencia desde sus tags
# ==========================================================================
def leer_raster(path):
    im = Image.open(path)
    arr = np.array(im, dtype="float64")
    tags = im.tag_v2

    nodata = -9999.0
    if 42113 in tags:  # GDAL_NODATA
        try:
            nodata = float(tags[42113])
        except (TypeError, ValueError):
            pass
    arr[arr == nodata] = np.nan

    if 33550 not in tags or 33922 not in tags:
        sys.exit(f"'{path}' no tiene etiquetas de georreferencia GeoTIFF estándar (33550/33922).")
    xscale, yscale, _ = tags[33550]
    i0, j0, k0, x0, y0, z0 = tuple(tags[33922])[:6]
    origin_x = x0 - i0 * xscale
    origin_y = y0 + j0 * yscale
    return arr, origin_x, origin_y, xscale, yscale


def valor_en(arr, origin_x, origin_y, xscale, yscale, lon, lat):
    nrows, ncols = arr.shape
    col = int((lon - origin_x) / xscale)
    row = int((origin_y - lat) / yscale)
    if 0 <= row < nrows and 0 <= col < ncols:
        return arr[row, col]
    return np.nan


# ==========================================================================
# 3) Recorrido de la carpeta por año y extracción
# ==========================================================================
PATRON_ARCHIVO = re.compile(r"([A-Za-z0-9]+)_(\d{4})_(\d{2})\.tif{1,2}$", re.IGNORECASE)


def listar_tifs(carpeta_raiz):
    """Devuelve {anio: [(mes, variable, ruta), ...]}, buscando en subcarpetas por año
    y, por si acaso, también directo en la carpeta raíz."""
    por_anio = {}
    rutas = glob.glob(os.path.join(carpeta_raiz, "**", "*.tif*"), recursive=True)
    for ruta in rutas:
        m = PATRON_ARCHIVO.search(os.path.basename(ruta))
        if not m:
            continue
        variable, anio, mes = m.group(1), int(m.group(2)), m.group(3)
        por_anio.setdefault(anio, []).append((mes, variable, ruta))
    return por_anio


def pedir_carpeta_tifs():
    while True:
        ruta = input("Carpeta raíz con los .tif (con subcarpetas por año, o sueltos): ").strip().strip('"').strip("'")
        if os.path.isdir(ruta):
            return ruta
        print(f"  No encontré la carpeta '{ruta}'. Intenta de nuevo.\n")


def pedir_ubicacion_parcelas():
    while True:
        modo = input("¿Tienes el shapefile (.shp) o un CSV de centroides? [S = shp / C = csv]: ").strip().lower()
        if modo.startswith("s"):
            ruta = input("Ruta del .shp (debe tener su .dbf al lado): ").strip().strip('"').strip("'")
            if os.path.isfile(ruta):
                return ruta, None
            print(f"  No encontré '{ruta}'.\n")
        elif modo.startswith("c"):
            ruta = input("Ruta del CSV con columnas ID_POLIGONO, lon, lat: ").strip().strip('"').strip("'")
            if os.path.isfile(ruta):
                return None, ruta
            print(f"  No encontré '{ruta}'.\n")
        else:
            print("  Escribe 'S' para shapefile o 'C' para CSV de centroides.\n")


def menu_interactivo():
    print("\n" + "=" * 70)
    print("  EXTRACTOR DE RASTERS POR PARCELA — RETO AGROCEBADA")
    print("=" * 70)
    while True:
        carpeta = pedir_carpeta_tifs()
        shp, centroides = pedir_ubicacion_parcelas()
        salida = input("Carpeta donde guardar los Excel [Enter = la actual]: ").strip().strip('"').strip("'") or "."

        try:
            ejecutar(carpeta, shp, centroides, salida)
        except SystemExit as e:
            print(f"\n  {e}")

        otra = input("\n¿Procesar otra carpeta de rasters? [s/N]: ").strip().lower()
        if not otra.startswith("s"):
            print("Listo, hasta luego.")
            break


# ==========================================================================
def ejecutar(carpeta_tifs, shp, centroides, salida):
    os.makedirs(salida, exist_ok=True)
    parcelas = cargar_parcelas(shp, centroides)
    print(f"Parcelas cargadas: {len(parcelas)}")

    por_anio = listar_tifs(carpeta_tifs)
    if not por_anio:
        sys.exit(f"No encontré archivos .tif con el patrón NOMBRE_AAAA_MM.tif dentro de '{carpeta_tifs}'.")

    for anio in sorted(por_anio):
        archivos = sorted(por_anio[anio])
        variables = sorted({v for _, v, _ in archivos})
        print(f"\nAño {anio}: {len(archivos)} archivo(s), variable(s) detectada(s): {variables}")

        tabla = parcelas.copy().set_index("ID_POLIGONO")
        for mes, variable, ruta in archivos:
            arr, ox, oy, xs, ys = leer_raster(ruta)
            col_nombre = f"{variable}_{mes}"
            tabla[col_nombre] = [
                valor_en(arr, ox, oy, xs, ys, lon, lat)
                for lon, lat in zip(parcelas.lon, parcelas.lat)
            ]
            n_validos = tabla[col_nombre].notna().sum()
            print(f"  {os.path.basename(ruta):26s} -> columna '{col_nombre}', "
                  f"{n_validos}/{len(tabla)} parcelas con valor")

        tabla = tabla.reset_index()
        nombre_var = variables[0] if len(variables) == 1 else "multivariable"
        ruta_salida = os.path.join(salida, f"{nombre_var}_{anio}.xlsx")
        tabla.to_excel(ruta_salida, index=False)
        print(f"  -> Guardado: {ruta_salida}  ({tabla.shape[0]} filas x {tabla.shape[1]} columnas)")

    print("\nListo. Cada Excel tiene una fila por parcela y una columna por variable-mes.")
    print("El siguiente paso es que decidas qué filtrar/agregar (p.ej. promedio jun-sep).")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tifs", default=None, help="Carpeta raíz con subcarpetas por año (2022/, 2023/, ...)")
    grupo = ap.add_mutually_exclusive_group()
    grupo.add_argument("--shp", help="Ruta al .shp de las parcelas (debe tener su .dbf al lado)")
    grupo.add_argument("--centroides", help="CSV alterno con columnas ID_POLIGONO, lon, lat")
    ap.add_argument("--salida", default=".", help="Carpeta donde guardar los .xlsx (por defecto la actual)")
    args = ap.parse_args()

    # Modo directo: si ya viene todo por línea de comandos, corre sin preguntar nada
    if args.tifs and (args.shp or args.centroides):
        ejecutar(args.tifs, args.shp, args.centroides, args.salida)
        return

    # Modo interfaz: pregunta lo que falte, paso a paso
    try:
        menu_interactivo()
    except (KeyboardInterrupt, EOFError):
        print("\nInterrumpido por el usuario. Hasta luego.")


if __name__ == "__main__":
    main()