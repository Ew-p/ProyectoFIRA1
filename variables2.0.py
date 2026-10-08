#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CONSTRUCTOR DE VARIABLES v2 — RETO AGROCEBADA
Línea base histórica (2022-2024) + anomalías vs 2025 + Dataset PRO (Planet 2025)

QUÉ HACE
--------
El rendimiento es del ciclo abril-octubre 2025. Este script ya NO promedia
2022-2025 como el pipeline anterior. En su lugar:

  1. BÁSICO (Sentinel-2 + Landsat), separado en dos partes:
     a) LÍNEA BASE (2022-2024): promedio y desv. estándar histórica de cada
        índice-mes-estadístico -> "qué es normal para ESTA parcela".
     b) ACTUAL (2025, filtrado a abril-octubre): promedio de cada
        índice-mes-estadístico -> "qué pasó este ciclo".
     c) ANOMALÍA = (actual - promedio histórico) / desv. std histórica.
        Si no hay línea base para un mes (std=0 o sin datos), la anomalía
        queda vacía (NaN) y se imputa después, no se inventa un número.
  2. PRO (Planet, ya es solo 2025): igual que "actual" arriba, filtrado a
     abril-octubre. Estos valores no se comparan contra línea base porque
     no hay Planet de años anteriores.
  3. Filtra nubosidad > umbral (por defecto 20%) en ambos datasets.
  4. Pega AREA_HA, RENDIMIENTO_T_HA, CONJUNTO.
  5. Guarda train.xlsx / prediccion.xlsx listos para xgb_seleccion_prediccion.py

USO
---
Modo con menú:
    python3 construir_variables_v2.py

Modo directo:
    python3 construir_variables_v2.py \
        --basico Conjunto_datos_BASICO_AgroCebada2026.csv \
        --pro Conjunto_datos_PRO_AgroCebada.csv \
        --area ID_area_rendimiento_70_30_Reto_AgroCebada.csv \
        --umbral 20 --mes-ini 4 --mes-fin 10 --salida ./variables_v2
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd


# ============================================================
# 1) Lectura flexible
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


def detectar_indices(columnas, sufijos=("_promedio", "_std", "_max", "_min")):
    """Encuentra los nombres base de índice (ej. 'ndvi') a partir de columnas
    tipo 'ndvi_promedio', 'ndvi_std', etc."""
    base = set()
    for c in columnas:
        for suf in sufijos:
            if c.endswith(suf):
                base.add(c[: -len(suf)])
    return sorted(base)


# ============================================================
# 2) Preparar un dataset (Básico o PRO): fecha, mes, año, filtro de nubosidad
# ============================================================
def preparar_dataset(df, umbral):
    fecha = pd.to_datetime(df["fecha_captura"], format="%d/%m/%Y", errors="coerce")
    if fecha.isna().mean() > 0.5:
        fecha = pd.to_datetime(df["fecha_captura"], errors="coerce")
    df = df.copy()
    df["_fecha"] = fecha
    df["_anio"] = fecha.dt.year
    df["_mes"] = fecha.dt.month

    indices = detectar_indices(df.columns)
    cols_valor = [c for i in indices for c in (f"{i}_promedio", f"{i}_std", f"{i}_max", f"{i}_min") if c in df.columns]

    df = df[df[cols_valor].notna().any(axis=1)]           # ya sin filas 100% vacías
    df = df[df["porcentaje_nubosidad"] <= umbral]          # filtro de nubosidad
    return df, indices, cols_valor


# ============================================================
# 3) Climatología mensual (promedio de las observaciones de cada mes)
# ============================================================
def climatologia_mensual(df, cols_valor, anios=None, meses=None, prefijo=""):
    """Una fila por parcela; columnas <col>_m<MM>. Si 'anios' se da, primero
    promedia por año dentro de cada mes (para no darle más peso a un año con
    más escenas) y luego promedia entre años."""
    d = df.copy()
    if anios is not None:
        d = d[d["_anio"].isin(anios)]
    if meses is not None:
        d = d[d["_mes"].isin(meses)]

    tabla = pd.DataFrame(index=sorted(d.ID_POLIGONO.unique()))
    tabla.index.name = "ID_POLIGONO"
    for c in cols_valor:
        if anios is not None and d["_anio"].nunique() > 1:
            paso1 = d.groupby(["ID_POLIGONO", "_anio", "_mes"])[c].mean()
            piv = paso1.groupby(["ID_POLIGONO", "_mes"]).mean().unstack()
        else:
            piv = d.groupby(["ID_POLIGONO", "_mes"])[c].mean().unstack()
        piv.columns = [f"{prefijo}{c}_m{int(m):02d}" for m in piv.columns]
        tabla = tabla.join(piv)
    return tabla


def desviacion_historica_mensual(df, cols_valor, anios, meses, prefijo=""):
    """Desv. estándar ENTRE AÑOS de cada mes (para poder calcular anomalías).
    Primero colapsa cada año-mes a un solo valor (promedio de esa ventana),
    luego calcula qué tanto varían esos valores anuales entre sí."""
    d = df[df["_anio"].isin(anios) & df["_mes"].isin(meses)]
    tabla = pd.DataFrame(index=sorted(d.ID_POLIGONO.unique()))
    tabla.index.name = "ID_POLIGONO"
    for c in cols_valor:
        paso1 = d.groupby(["ID_POLIGONO", "_anio", "_mes"])[c].mean()
        sd = paso1.groupby(["ID_POLIGONO", "_mes"]).std().unstack()
        sd.columns = [f"{prefijo}{c}_m{int(m):02d}_sd" for m in sd.columns]
        tabla = tabla.join(sd)
    return tabla


# ============================================================
# 4) Flujo principal
# ============================================================
def ejecutar(ruta_basico, ruta_pro, ruta_area, umbral, mes_ini, mes_fin, salida_dir):
    os.makedirs(salida_dir, exist_ok=True)
    meses_ciclo = list(range(mes_ini, mes_fin + 1))
    anios_base = [2022, 2023, 2024]

    print("=" * 70)
    print("1) BÁSICO (Sentinel-2 + Landsat)")
    print("=" * 70)
    basico = leer_tabla(ruta_basico)
    basico, idx_basico, cols_basico = preparar_dataset(basico, umbral)
    print(f"Filas tras filtrar nubosidad > {umbral}%: {len(basico)}")
    print(f"Índices detectados en Básico: {len(idx_basico)} -> {idx_basico}")

    print("\n-- 1a) Línea base histórica (2022-2024, ciclo abr-oct) --")
    base_clima = climatologia_mensual(basico, cols_basico, anios=anios_base, meses=meses_ciclo, prefijo="base_")
    base_sd = desviacion_historica_mensual(basico, cols_basico, anios_base, meses_ciclo, prefijo="base_")
    print(f"  Parcelas con línea base: {base_clima.notna().any(axis=1).sum()}  |  columnas: {base_clima.shape[1]}")

    print("\n-- 1b) Actual (2025, ciclo abr-oct) --")
    actual_clima = climatologia_mensual(basico, cols_basico, anios=[2025], meses=meses_ciclo, prefijo="act_")
    print(f"  Parcelas con dato 2025: {actual_clima.notna().any(axis=1).sum()}  |  columnas: {actual_clima.shape[1]}")

    print("\n-- 1c) Anomalía = (actual_2025 - base_promedio) / base_desv_std --")
    anom = pd.DataFrame(index=sorted(basico.ID_POLIGONO.unique()))
    anom.index.name = "ID_POLIGONO"
    columnas_anom = {}
    for c in cols_basico:
        for m in meses_ciclo:
            col_act = f"act_{c}_m{m:02d}"
            col_base = f"base_{c}_m{m:02d}"
            col_sd = f"base_{c}_m{m:02d}_sd"
            if col_act in actual_clima.columns and col_base in base_clima.columns and col_sd in base_sd.columns:
                a = actual_clima[col_act]
                b = base_clima[col_base]
                s = base_sd[col_sd].replace(0, np.nan)
                columnas_anom[f"anom_{c}_m{m:02d}"] = (a - b) / s
    anom = pd.concat(columnas_anom, axis=1) if columnas_anom else anom
    n_anom = len(columnas_anom)
    print(f"  Columnas de anomalía construidas: {n_anom}")
    print(f"  % de valores no calculables (sin línea base o std=0): "
          f"{anom.isna().mean().mean() * 100:.1f}%")

    print("\n" + "=" * 70)
    print("2) PRO (Planet, ya es solo 2025)")
    print("=" * 70)
    pro = leer_tabla(ruta_pro)
    pro, idx_pro, cols_pro = preparar_dataset(pro, umbral)
    print(f"Filas tras filtrar nubosidad > {umbral}%: {len(pro)}")
    print(f"Índices detectados en PRO: {idx_pro}")
    pro_clima = climatologia_mensual(pro, cols_pro, anios=[2025], meses=meses_ciclo, prefijo="pro_")
    print(f"  Parcelas con dato PRO 2025: {pro_clima.notna().any(axis=1).sum()}  |  columnas: {pro_clima.shape[1]}")

    print("\n" + "=" * 70)
    print("3) UNIENDO TODO + AREA/RENDIMIENTO")
    print("=" * 70)
    tabla = base_clima.join(actual_clima, how="outer").join(anom, how="outer").join(pro_clima, how="outer")
    print(f"Tabla combinada: {tabla.shape[0]} filas x {tabla.shape[1]} columnas")

    area = leer_tabla(ruta_area)
    col_id_area = next((c for c in area.columns if "polig" in c.lower() or c.lower() == "id"), "ID_POLIGONO")
    col_area = next((c for c in area.columns if "area" in c.lower()), None)
    col_rend = next((c for c in area.columns if "rendim" in c.lower()), None)
    col_conj = next((c for c in area.columns if "conjunto" in c.lower()), None)
    area = area.rename(columns={col_id_area: "ID_POLIGONO", col_area: "AREA_HA",
                                 col_rend: "RENDIMIENTO_T_HA", col_conj: "CONJUNTO"})
    tabla = tabla.reset_index().merge(area[["ID_POLIGONO", "AREA_HA", "RENDIMIENTO_T_HA", "CONJUNTO"]],
                                       on="ID_POLIGONO", how="left")

    sin_conjunto = tabla.CONJUNTO.isna().sum()
    if sin_conjunto:
        print(f"AVISO: {sin_conjunto} parcela(s) sin CONJUNTO conocido, se descartan.")
    tabla = tabla.dropna(subset=["CONJUNTO"])

    train = tabla[tabla.CONJUNTO == "ENTRENAMIENTO"].drop(columns=["CONJUNTO"])
    pred = tabla[tabla.CONJUNTO == "PREDICCION"].drop(columns=["CONJUNTO", "RENDIMIENTO_T_HA"])

    ruta_train = os.path.join(salida_dir, "v2_train.xlsx")
    ruta_pred = os.path.join(salida_dir, "v2_prediccion.xlsx")
    train.to_excel(ruta_train, index=False)
    pred.to_excel(ruta_pred, index=False)

    print("\n" + "=" * 70)
    print(f"Guardado TRAIN      -> {ruta_train}   ({train.shape[0]} filas, {train.shape[1]} columnas)")
    print(f"Guardado PREDICCION -> {ruta_pred}   ({pred.shape[0]} filas, {pred.shape[1]} columnas)")
    print("=" * 70)
    print("Bloques de columnas incluidos: base_* (línea base 2022-24), "
          "act_* (Sentinel/Landsat 2025), anom_* (anomalía), pro_* (Planet 2025)")
    print(f"\nListos para xgb_seleccion_prediccion.py:")
    print(f"  python3 xgb_seleccion_prediccion.py --train {ruta_train} --prediccion {ruta_pred} "
          f"--target RENDIMIENTO_T_HA --id ID_POLIGONO")


# ============================================================
# 5) Interfaz / argumentos
# ============================================================
def menu_interactivo():
    print("=" * 70)
    print("  CONSTRUCTOR DE VARIABLES v2 (línea base + anomalía + PRO)")
    print("=" * 70)
    ruta_basico = input("Ruta del Dataset BÁSICO (Conjunto_datos_BASICO...csv): ").strip().strip('"').strip("'")
    ruta_pro = input("Ruta del Dataset PRO (Conjunto_datos_PRO...csv): ").strip().strip('"').strip("'")
    ruta_area = input("Ruta del CSV/XLSX de ID_area_rendimiento: ").strip().strip('"').strip("'")
    u_txt = input("Umbral de nubosidad [Enter = 20]: ").strip()
    umbral = float(u_txt) if u_txt else 20.0
    mi_txt = input("Mes de inicio del ciclo [Enter = 4 (abril)]: ").strip()
    mes_ini = int(mi_txt) if mi_txt else 4
    mf_txt = input("Mes de fin del ciclo [Enter = 10 (octubre)]: ").strip()
    mes_fin = int(mf_txt) if mf_txt else 10
    salida = input("Carpeta de salida [Enter = ./variables_v2]: ").strip() or "./variables_v2"
    ejecutar(ruta_basico, ruta_pro, ruta_area, umbral, mes_ini, mes_fin, salida)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--basico")
    ap.add_argument("--pro")
    ap.add_argument("--area")
    ap.add_argument("--umbral", type=float, default=20.0)
    ap.add_argument("--mes-ini", type=int, default=4)
    ap.add_argument("--mes-fin", type=int, default=10)
    ap.add_argument("--salida", default="./variables_v2")
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