r"""
Igual que armar_dataset_final.py, pero le das UNA SOLA carpeta raíz y él solo
descubre, dentro de todas sus subcarpetas, cuáles archivos son:
  - el shapefile de parcelas (.shp con Municipio/lon/lat)
  - el CSV de área/rendimiento/conjunto
  - los archivos satelitales (los que tengan ID + fecha + nubosidad,
    sin importar cuántos sean ni cómo se llamen)
Ignora automáticamente lo que no reconoce (topografía, clima, reportes que
el propio proyecto ya generó, etc.) y te dice qué clasificó como qué antes
de construir el dataset, para que puedas frenarlo si detecta algo raro.

USO
---
python3 armar_dataset_desde_carpeta.py --carpeta "C:\...\ProyectoFIRA1" --umbral 20 --salida ./dataset_final

Si lo corres sin argumentos, te pregunta la carpeta paso a paso.
"""
import argparse
import glob
import os
import struct
import sys

import numpy as np
import pandas as pd

MESES_CLAVE = {
    "veg_prim": {"savi": [3, 4, 5, 6], "lai": [3, 4, 5, 6], "evi": [3, 4, 5, 6]},
    "ndvi_inv": {"ndvi": [1, 2]},
    "ndti_jun": {"ndti": [6]},
}
INDICES_NECESARIOS = ["ndvi", "savi", "lai", "evi", "ndti"]
IGNORAR_PREFIJOS = ("dataset_final_",)
IGNORAR_SUFIJOS = ("_resumen_parcela.csv", "_resumen_mes_anio.csv", "_huecos.csv")


# ==========================================================================
# 1) Lectura flexible de tablas (csv con varias codificaciones, o xlsx)
# ==========================================================================
def leer_tabla(path):
    ext = path.rsplit(".", 1)[-1].lower()
    if ext in ("xlsx", "xls"):
        return pd.read_excel(path)
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def detectar_col(df, *patrones):
    for c in df.columns:
        if any(p in c.lower() for p in patrones):
            return c
    return None


# ==========================================================================
# 2) Shapefile mínimo (sin geopandas): Municipio, Estado, centroide
# ==========================================================================
def _read_dbf(path):
    b = open(path, "rb").read()
    n, hl, rl = struct.unpack("<xxxxIHH", b[:12])
    fields, off = [], 32
    while b[off] != 0x0D:
        nm = b[off:off + 11].split(b"\x00")[0].decode()
        t = chr(b[off + 11])
        ln = b[off + 16]
        fields.append((nm, t, ln))
        off += 32
    rows = []
    for i in range(n):
        r = b[hl + i * rl:hl + (i + 1) * rl]
        pos, row = 1, {}
        for nm, t, ln in fields:
            v = r[pos:pos + ln].decode("utf-8", "ignore").strip()
            pos += ln
            row[nm] = (float(v) if v not in ("", "*" * ln) else np.nan) if t in "NF" else v
        rows.append(row)
    return pd.DataFrame(rows)


def _read_shp_centroids(path):
    b = open(path, "rb").read()
    pos, centroids = 100, []
    while pos < len(b):
        _, cl = struct.unpack(">ii", b[pos:pos + 8])
        rec = b[pos + 8:pos + 8 + cl * 2]
        pos += 8 + cl * 2
        if struct.unpack("<i", rec[:4])[0] == 0:
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


def cargar_shapefile(shp_path):
    base = shp_path.rsplit(".", 1)[0]
    df = _read_dbf(base + ".dbf")
    df["lon"], df["lat"] = zip(*_read_shp_centroids(base + ".shp"))
    id_col = next((c for c in df.columns if "polig" in c.lower() or c.lower() == "id"), None)
    if id_col:
        df = df.rename(columns={id_col: "ID_POLIGONO"})
    return df


# ==========================================================================
# 3) Satélite: cargar varios archivos, filtrar nubosidad, agregar por parcela
# ==========================================================================
def cargar_satelite(rutas, umbral):
    piezas = []
    for r in rutas:
        d = leer_tabla(r)
        col_id = detectar_col(d, "polig", "id_polig") or detectar_col(d, "^id$")
        col_fecha = detectar_col(d, "fecha")
        col_nub = detectar_col(d, "nub")
        if not (col_id and col_fecha and col_nub):
            print(f"  Aviso: '{r}' no tiene columnas de ID/fecha/nubosidad reconocibles, se omite.")
            continue
        d = d.rename(columns={col_id: "ID_POLIGONO", col_fecha: "_fecha_txt", col_nub: "_nub"})
        fecha = pd.to_datetime(d["_fecha_txt"], format="%d/%m/%Y", errors="coerce")
        if fecha.isna().mean() > 0.5:
            fecha = pd.to_datetime(d["_fecha_txt"], errors="coerce")
        d["_mes"] = fecha.dt.month
        cols_idx = {}
        for idx in INDICES_NECESARIOS:
            c = next((c for c in d.columns if c.lower() == f"{idx}_promedio"), None)
            if c:
                cols_idx[idx] = c
        piezas.append(d[["ID_POLIGONO", "_mes", "_nub"] + list(cols_idx.values())].rename(columns={v: k for k, v in cols_idx.items()}))
        print(f"  Leído '{os.path.basename(r)}': {len(d)} filas, índices encontrados: {list(cols_idx)}")

    todo = pd.concat(piezas, ignore_index=True)
    antes = len(todo)
    todo = todo[todo["_nub"] <= umbral]
    print(f"  Filas satelitales tras filtrar nubosidad > {umbral}%: {len(todo)} de {antes}")

    # promedio mensual por parcela, para cada índice
    F = pd.DataFrame(index=sorted(todo.ID_POLIGONO.unique()))
    for idx in INDICES_NECESARIOS:
        if idx not in todo.columns:
            continue
        piv = todo.groupby(["ID_POLIGONO", "_mes"])[idx].mean().unstack()
        piv.columns = [f"{idx}_m{int(c):02d}" for c in piv.columns]
        F = F.join(piv)
    return F


def calcular_variables_compactas(F):
    def cols(idx, meses):
        return [f"{idx}_m{m:02d}" for m in meses if f"{idx}_m{m:02d}" in F.columns]

    C = pd.DataFrame(index=F.index)
    z = lambda s: (s - s.mean()) / s.std()
    partes = []
    for idx, meses in MESES_CLAVE["veg_prim"].items():
        cs = cols(idx, meses)
        if cs:
            partes.append(F[cs].apply(z))
    C["veg_prim"] = pd.concat(partes, axis=1).mean(axis=1) if partes else np.nan

    cs = cols("ndvi", MESES_CLAVE["ndvi_inv"]["ndvi"])
    C["ndvi_inv"] = F[cs].mean(axis=1) if cs else np.nan

    cs = cols("ndti", MESES_CLAVE["ndti_jun"]["ndti"])
    C["ndti_jun"] = F[cs].mean(axis=1) if cs else np.nan
    return C


# ==========================================================================
# 3.5) Escaneo automático de la carpeta raíz
# ==========================================================================
def _nombre_ignorable(ruta):
    nombre = os.path.basename(ruta)
    return nombre.startswith(IGNORAR_PREFIJOS) or nombre.endswith(IGNORAR_SUFIJOS)


def escanear_carpeta(raiz):
    shp_candidatos = sorted(glob.glob(os.path.join(raiz, "**", "*.shp"), recursive=True))
    tabla_candidatos = [
        r for ext in ("*.csv", "*.xlsx", "*.xls")
        for r in glob.glob(os.path.join(raiz, "**", ext), recursive=True)
        if not _nombre_ignorable(r)
    ]

    satelite, rendimiento, sin_usar = [], None, []
    for ruta in tabla_candidatos:
        try:
            d = leer_tabla(ruta)
        except Exception as e:
            sin_usar.append((ruta, f"no se pudo leer ({e})"))
            continue

        col_id = detectar_col(d, "polig", "id_polig") or detectar_col(d, "^id$")
        col_fecha = detectar_col(d, "fecha")
        col_nub = detectar_col(d, "nub")
        col_rend = detectar_col(d, "rendimiento")
        col_conj = detectar_col(d, "conjunto")

        if col_id and col_fecha and col_nub:
            satelite.append(ruta)
        elif col_id and col_rend and col_conj:
            if rendimiento is None:
                rendimiento = ruta
            else:
                sin_usar.append((ruta, "parece otro archivo de rendimiento, se ignora (ya tengo uno)"))
        else:
            sin_usar.append((ruta, "no coincide con satélite ni con rendimiento"))

    if not shp_candidatos:
        shp = None
    elif len(shp_candidatos) == 1:
        shp = shp_candidatos[0]
    else:
        # si hay varios, prioriza uno que tenga pinta de "parcelas"
        con_parcela = [s for s in shp_candidatos if "parcela" in s.lower()]
        shp = con_parcela[0] if con_parcela else shp_candidatos[0]
        extra = [s for s in shp_candidatos if s != shp]
        for s in extra:
            sin_usar.append((s, "otro .shp encontrado, no se usó (había uno con pinta de parcelas)"))

    return shp, rendimiento, satelite, sin_usar


def mostrar_hallazgos(shp, rendimiento, satelite, sin_usar):
    print("\n" + "-" * 70)
    print("LO QUE ENCONTRÉ EN LA CARPETA")
    print("-" * 70)
    print(f"Shapefile de parcelas: {shp or '¡NO ENCONTRADO!'}")
    print(f"CSV de rendimiento:    {rendimiento or '¡NO ENCONTRADO!'}")
    print(f"Archivos satelitales ({len(satelite)}):")
    for s in satelite:
        print(f"   - {s}")
    if sin_usar:
        print(f"\nArchivos que encontré pero NO usé ({len(sin_usar)}):")
        for r, motivo in sin_usar:
            print(f"   - {os.path.basename(r)}: {motivo}")
    print()


# ==========================================================================
# 4) Ensamble final
# ==========================================================================
def armar(rutas_satelite, shp_path, rendimiento_path, umbral, salida):
    os.makedirs(salida, exist_ok=True)

    print("\n[1/4] Cargando y agregando datos satelitales...")
    F = cargar_satelite(rutas_satelite, umbral)
    C = calcular_variables_compactas(F)

    print("\n[2/4] Cargando shapefile de parcelas...")
    parc = cargar_shapefile(shp_path).set_index("ID_POLIGONO")
    cols_shp = [c for c in ["Municipio", "Estado", "lon", "lat"] if c in parc.columns]

    print("[3/4] Cargando rendimiento/área/conjunto...")
    rend = leer_tabla(rendimiento_path)
    rend = rend.loc[:, ~rend.columns.str.match(r"^Unnamed")]  # limpia columnas vacías por comas colgantes
    col_id = detectar_col(rend, "polig", "id_polig") or detectar_col(rend, "^id$")
    rend = rend.rename(columns={col_id: "ID_POLIGONO"}).set_index("ID_POLIGONO")

    print("[4/4] Uniendo todo y guardando...")
    D = rend.join(parc[cols_shp]).join(C)
    D = D.reset_index()

    col_conjunto = next((c for c in D.columns if "conjunto" in c.lower()), None)
    if col_conjunto is None:
        sys.exit("No encontré una columna 'CONJUNTO' en el archivo de rendimiento.")

    ruta_train = os.path.join(salida, "dataset_final_entrenamiento.csv")
    ruta_pred = os.path.join(salida, "dataset_final_prediccion.csv")
    D[D[col_conjunto].str.upper().str.startswith("ENTR")].to_csv(ruta_train, index=False)
    D[D[col_conjunto].str.upper().str.startswith("PRED")].to_csv(ruta_pred, index=False)

    print(f"\nListo:")
    print(f"  {ruta_train}  ({(D[col_conjunto].str.upper().str.startswith('ENTR')).sum()} filas)")
    print(f"  {ruta_pred}   ({(D[col_conjunto].str.upper().str.startswith('PRED')).sum()} filas)")
    print(f"\nColumnas: {list(D.columns)}")
    faltan = D[cols_shp + ["veg_prim", "ndvi_inv", "ndti_jun"]].isna().sum()
    if faltan.sum():
        print("\nValores faltantes por columna (revisa si esperabas alguno):")
        print(faltan[faltan > 0].to_string())


# ==========================================================================
def pedir_carpeta():
    while True:
        raiz = input("Carpeta RAÍZ del proyecto (busco todo dentro, en subcarpetas): ").strip().strip('"').strip("'")
        if os.path.isdir(raiz):
            return raiz
        print(f"  No encontré la carpeta '{raiz}'. Intenta de nuevo.\n")


def menu_interactivo():
    print("\n" + "=" * 70)
    print("  ARMAR DATASET FINAL (AUTO) — RETO AGROCEBADA")
    print("=" * 70)
    while True:
        raiz = pedir_carpeta()
        print("\nBuscando shapefile, rendimiento y archivos satelitales...")
        shp, rendimiento, satelite, sin_usar = escanear_carpeta(raiz)
        mostrar_hallazgos(shp, rendimiento, satelite, sin_usar)

        if not (shp and rendimiento and satelite):
            print("  Me falta al menos uno de los tres (shapefile / rendimiento / satelitales).")
            print("  Revisa la carpeta o dame otra ruta.\n")
            continue

        ok = input("¿Procedo a armar el dataset con esto? [S/n]: ").strip().lower()
        if ok.startswith("n"):
            print("  Cancelado. Puedes darme otra carpeta.\n")
            continue

        u_txt = input("Umbral de nubosidad a aplicar [Enter = 20]: ").strip()
        umbral = float(u_txt) if u_txt else 20.0
        sal = input("Carpeta de salida [Enter = ./dataset_final]: ").strip().strip('"').strip("'") or "./dataset_final"
        armar(satelite, shp, rendimiento, umbral, sal)

        otra = input("\n¿Escanear otra carpeta? [s/N]: ").strip().lower()
        if not otra.startswith("s"):
            print("Listo, hasta luego.")
            break


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--carpeta", default=None, help="Carpeta raíz del proyecto (busca todo dentro, recursivo)")
    ap.add_argument("--umbral", type=float, default=20.0, help="Umbral de nubosidad a filtrar (%%, default 20)")
    ap.add_argument("--salida", default="./dataset_final", help="Carpeta de salida")
    ap.add_argument("--forzar", action="store_true", help="No pedir confirmación, arma directo con lo que encuentre")
    args = ap.parse_args()

    if args.carpeta:
        print(f"Buscando en: {args.carpeta}")
        shp, rendimiento, satelite, sin_usar = escanear_carpeta(args.carpeta)
        mostrar_hallazgos(shp, rendimiento, satelite, sin_usar)
        if not (shp and rendimiento and satelite):
            sys.exit("Falta el shapefile, el CSV de rendimiento o algún archivo satelital. Revisa la carpeta.")
        if not args.forzar:
            ok = input("¿Procedo a armar el dataset con esto? [S/n]: ").strip().lower()
            if ok.startswith("n"):
                sys.exit("Cancelado por el usuario.")
        armar(satelite, shp, rendimiento, args.umbral, args.salida)
        return

    try:
        menu_interactivo()
    except (KeyboardInterrupt, EOFError):
        print("\nInterrumpido por el usuario. Hasta luego.")


if __name__ == "__main__":
    main()