#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CONDENSADOR DE SERIES TEMPORALES — RETO AGROCEBADA

En vez de una columna por mes (lo que infla a miles de características),
cada índice se resume con estadísticos de la serie completa del ciclo
2025 (abril-octubre), igual que se condensan vectores de características
en visión por computadora (pooling):

  media, desv_std, min, max, mediana, skew (asimetría),
  kurtosis (qué tan picuda/plana es la distribución),
  tendencia (pendiente de una regresión lineal valor~tiempo,
             ¿subía o bajaba el índice durante el ciclo?),
  n_obs (cuántas observaciones válidas tuvo esa parcela)

Se hace para:
  - BÁSICO 2025 (Sentinel-2 + Landsat), ventana abril-octubre
  - PRO 2025 (Planet), misma ventana
  - Anomalía de la media y la std vs. línea base 2022-2024 (BÁSICO),
    SOLO 2 estadísticos por índice (no los 8), para no reintroducir
    el mismo problema de redundancia de la versión anterior.

Con 23 índices Básico + 4 PRO = 27 índices:
  27 x 8 estadísticos (actual)      = 216 columnas
  23 x 2 estadísticos (anomalía)    =  46 columnas
  Total aprox.                      = 262 columnas  (antes: 2,044)

USO
---
Modo con menú:
    python3 condensar_series.py

Modo directo:
    python3 condensar_series.py \
        --basico Conjunto_datos_BASICO_AgroCebada2026.csv \
        --pro Conjunto_datos_PRO_AgroCebada.csv \
        --area ID_area_rendimiento_70_30_Reto_AgroCebada.csv \
        --umbral 20 --mes-ini 4 --mes-fin 10 --salida ./condensado
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis


# ============================================================
# 1) Lectura y preparación (igual que en los scripts anteriores)
# ============================================================
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


def detectar_indices(columnas):
    base = set()
    for c in columnas:
        if c.endswith("_promedio"):
            base.add(c[: -len("_promedio")])
    return sorted(base)


def preparar_dataset(df, umbral):
    fecha = pd.to_datetime(df["fecha_captura"], format="%d/%m/%Y", errors="coerce")
    if fecha.isna().mean() > 0.5:
        fecha = pd.to_datetime(df["fecha_captura"], errors="coerce")
    df = df.copy()
    df["_fecha"] = fecha
    df["_anio"] = fecha.dt.year
    df["_mes"] = fecha.dt.month
    df["_dia_ciclo"] = (fecha - fecha.dt.to_period("Y").dt.start_time).dt.days  # día del año, para la tendencia

    indices = detectar_indices(df.columns)
    cols_prom = [f"{i}_promedio" for i in indices if f"{i}_promedio" in df.columns]

    df = df[df[cols_prom].notna().any(axis=1)]
    df = df[df["porcentaje_nubosidad"] <= umbral]
    return df, indices, cols_prom


# ============================================================
# 2) Condensación: una fila por parcela, estadísticos por índice
# ============================================================
def _stats_de_serie(valores, dias):
    v = np.asarray(valores, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return dict(media=np.nan, std=np.nan, minimo=np.nan, maximo=np.nan,
                     mediana=np.nan, skew=np.nan, kurtosis=np.nan, tendencia=np.nan, n_obs=0)
    out = dict(
        media=float(np.mean(v)), std=float(np.std(v)) if len(v) > 1 else 0.0,
        minimo=float(np.min(v)), maximo=float(np.max(v)), mediana=float(np.median(v)),
        skew=float(skew(v)) if len(v) > 2 else np.nan,
        kurtosis=float(kurtosis(v)) if len(v) > 3 else np.nan,
        n_obs=len(v),
    )
    d = np.asarray(dias, dtype=float)
    mask = ~np.isnan(d) & ~np.isnan(v if len(v) == len(d) else np.full(len(d), np.nan))
    if len(v) >= 2 and len(d) == len(valores):
        dd = np.asarray(dias, dtype=float)
        vv = np.asarray(valores, dtype=float)
        ok = ~np.isnan(dd) & ~np.isnan(vv)
        if ok.sum() >= 2 and np.std(dd[ok]) > 0:
            pend = np.polyfit(dd[ok], vv[ok], 1)[0]
            out["tendencia"] = float(pend)
        else:
            out["tendencia"] = np.nan
    else:
        out["tendencia"] = np.nan
    return out


def condensar(df, cols_prom, meses, anios, prefijo):
    d = df[df["_mes"].isin(meses) & df["_anio"].isin(anios)]
    filas = {}
    for pid, g in d.groupby("ID_POLIGONO"):
        fila = {}
        for c in cols_prom:
            idx = c[: -len("_promedio")]
            st = _stats_de_serie(g[c].values, g["_dia_ciclo"].values)
            for k, val in st.items():
                fila[f"{prefijo}{idx}_{k}"] = val
        filas[pid] = fila
    tabla = pd.DataFrame.from_dict(filas, orient="index")
    tabla.index.name = "ID_POLIGONO"
    return tabla


# ============================================================
# 3) Flujo principal
# ============================================================
def ejecutar(ruta_basico, ruta_pro, ruta_area, umbral, mes_ini, mes_fin, salida_dir):
    os.makedirs(salida_dir, exist_ok=True)
    meses_ciclo = list(range(mes_ini, mes_fin + 1))

    print("=" * 70)
    print("1) BÁSICO (Sentinel-2 + Landsat) -- condensando 2025")
    print("=" * 70)
    basico = leer_tabla(ruta_basico)
    basico, idx_basico, cols_basico = preparar_dataset(basico, umbral)
    actual = condensar(basico, cols_basico, meses_ciclo, [2025], prefijo="act_")
    print(f"Índices Básico: {len(idx_basico)}  ->  columnas condensadas: {actual.shape[1]}")
    print(f"Parcelas con dato: {actual.shape[0]}")

    print("\n-- Línea base 2022-2024, SOLO media y std (para la anomalía) --")
    base = condensar(basico, cols_basico, meses_ciclo, [2022, 2023, 2024], prefijo="base_")
    anom = pd.DataFrame(index=actual.index)
    for idx in idx_basico:
        a_media = f"act_{idx}_media"
        b_media = f"base_{idx}_media"
        b_std = f"base_{idx}_std"
        if a_media in actual.columns and b_media in base.columns:
            m = actual[a_media]; bm = base.reindex(actual.index)[b_media]
            bs = base.reindex(actual.index)[b_std].replace(0, np.nan)
            anom[f"anom_{idx}_media"] = (m - bm) / bs
        a_std = f"act_{idx}_std"
        if a_std in actual.columns and b_std in base.columns:
            s = actual[a_std]; bs2 = base.reindex(actual.index)[b_std].replace(0, np.nan)
            anom[f"anom_{idx}_variabilidad"] = (s - bs2) / bs2
    print(f"Columnas de anomalía (solo media y variabilidad): {anom.shape[1]}")

    print("\n" + "=" * 70)
    print("2) PRO (Planet 2025) -- condensando")
    print("=" * 70)
    pro = leer_tabla(ruta_pro)
    pro, idx_pro, cols_pro = preparar_dataset(pro, umbral)
    actual_pro = condensar(pro, cols_pro, meses_ciclo, [2025], prefijo="pro_")
    print(f"Índices PRO: {idx_pro}  ->  columnas condensadas: {actual_pro.shape[1]}")

    print("\n" + "=" * 70)
    print("3) UNIENDO TODO + AREA/RENDIMIENTO")
    print("=" * 70)
    tabla = actual.join(anom, how="outer").join(actual_pro, how="outer")
    print(f"Tabla final: {tabla.shape[0]} filas x {tabla.shape[1]} columnas "
          f"(antes de esto: 2,044 columnas con el enfoque mensual)")

    area = leer_tabla(ruta_area)
    col_id_area = next((c for c in area.columns if "polig" in c.lower() or c.lower() == "id"), "ID_POLIGONO")
    col_area = next((c for c in area.columns if "area" in c.lower()), None)
    col_rend = next((c for c in area.columns if "rendim" in c.lower()), None)
    col_conj = next((c for c in area.columns if "conjunto" in c.lower()), None)
    area = area.rename(columns={col_id_area: "ID_POLIGONO", col_area: "AREA_HA",
                                 col_rend: "RENDIMIENTO_T_HA", col_conj: "CONJUNTO"})
    tabla = tabla.reset_index().merge(area[["ID_POLIGONO", "AREA_HA", "RENDIMIENTO_T_HA", "CONJUNTO"]],
                                       on="ID_POLIGONO", how="left")
    tabla = tabla.dropna(subset=["CONJUNTO"])

    train = tabla[tabla.CONJUNTO == "ENTRENAMIENTO"].drop(columns=["CONJUNTO"])
    pred = tabla[tabla.CONJUNTO == "PREDICCION"].drop(columns=["CONJUNTO", "RENDIMIENTO_T_HA"])

    ruta_train = os.path.join(salida_dir, "condensado_train.xlsx")
    ruta_pred = os.path.join(salida_dir, "condensado_prediccion.xlsx")
    train.to_excel(ruta_train, index=False)
    pred.to_excel(ruta_pred, index=False)

    print(f"\nGuardado TRAIN      -> {ruta_train}   ({train.shape[0]} filas, {train.shape[1]} columnas)")
    print(f"Guardado PREDICCION -> {ruta_pred}   ({pred.shape[0]} filas, {pred.shape[1]} columnas)")
    print(f"\nRazón filas/columnas en train: {train.shape[0] / (train.shape[1]-1):.2f} "
          f"(antes era 0.068 con el enfoque mensual)")
    print(f"\npython3 xgb_seleccion_prediccion.py --train {ruta_train} --prediccion {ruta_pred} "
          f"--target RENDIMIENTO_T_HA --id ID_POLIGONO")


# ============================================================
# 4) Interfaz
# ============================================================
def menu_interactivo():
    print("=" * 70)
    print("  CONDENSADOR DE SERIES TEMPORALES (media/std/min/max/skew/kurtosis/tendencia)")
    print("=" * 70)
    ruta_basico = input("Ruta del Dataset BÁSICO: ").strip().strip('"').strip("'")
    ruta_pro = input("Ruta del Dataset PRO: ").strip().strip('"').strip("'")
    ruta_area = input("Ruta del CSV/XLSX de ID_area_rendimiento: ").strip().strip('"').strip("'")
    u_txt = input("Umbral de nubosidad [Enter = 20]: ").strip()
    umbral = float(u_txt) if u_txt else 20.0
    mi_txt = input("Mes de inicio del ciclo [Enter = 4]: ").strip()
    mes_ini = int(mi_txt) if mi_txt else 4
    mf_txt = input("Mes de fin del ciclo [Enter = 10]: ").strip()
    mes_fin = int(mf_txt) if mf_txt else 10
    salida = input("Carpeta de salida [Enter = ./condensado]: ").strip() or "./condensado"
    ejecutar(ruta_basico, ruta_pro, ruta_area, umbral, mes_ini, mes_fin, salida)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--basico")
    ap.add_argument("--pro")
    ap.add_argument("--area")
    ap.add_argument("--umbral", type=float, default=20.0)
    ap.add_argument("--mes-ini", type=int, default=4)
    ap.add_argument("--mes-fin", type=int, default=10)
    ap.add_argument("--salida", default="./condensado")
    args = ap.parse_args()

    if args.basico and args.pro and args.area:
        ejecutar(args.basico, args.pro, args.area, args.umbral, args.mes_ini, args.mes_fin, args.salida)
    else:
        try:
            menu_interactivo()
        except (KeyboardInterrupt, EOFError):
            print("\nInterrumpido por el usuario.")


if __name__ == "__main__":
    main()