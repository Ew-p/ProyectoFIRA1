#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==========================================================================
 ANÁLISIS DE ROBUSTEZ Y VARIABILIDAD  ·  INTERFAZ DE TERMINAL
 1000 CORRIDAS DE K-FOLD CV CON XGBOOST + SELECCIÓN DE CARACTERÍSTICAS
==========================================================================

Mejoras clave:
  · 'TODAS' está EXCLUIDO por defecto del bucle masivo. Si lo quieres,
    se activa con --incluir-baseline y solo corre pocas veces (30).
  · Preselección opcional por Mutual Information + correlación cuando
    hay muchas columnas (recomendado con p >> n).
  · Interfaz de terminal con colores, banners, barras y menús.
==========================================================================
"""

import argparse
import os
import sys
import time
import shutil
import warnings

import numpy as np
import pandas as pd

from sklearn.feature_selection import (
    mutual_info_regression, SelectKBest, f_regression,
)
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor
except ImportError:
    sys.exit("\n[ERROR] Este script necesita xgboost.\n"
             "Instálalo con:  pip install xgboost\n")


# ============================================================
# 0) INTERFAZ DE TERMINAL
# ============================================================
class C:
    ENABLED = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None

    RESET   = "\033[0m"   if ENABLED else ""
    BOLD    = "\033[1m"   if ENABLED else ""
    DIM     = "\033[2m"   if ENABLED else ""
    RED     = "\033[31m"  if ENABLED else ""
    GREEN   = "\033[32m"  if ENABLED else ""
    YELLOW  = "\033[33m"  if ENABLED else ""
    BLUE    = "\033[34m"  if ENABLED else ""
    MAGENTA = "\033[35m"  if ENABLED else ""
    CYAN    = "\033[36m"  if ENABLED else ""
    WHITE   = "\033[37m"  if ENABLED else ""
    BG_BLUE = "\033[44m"  if ENABLED else ""


def _w():
    try:
        return shutil.get_terminal_size((100, 25)).columns
    except Exception:
        return 100


def clear():
    if C.ENABLED:
        os.system("cls" if os.name == "nt" else "clear")


def banner(titulo, subtitulo=""):
    w = _w()
    print()
    print(C.BG_BLUE + C.WHITE + C.BOLD + " " * w + C.RESET)
    print(C.BG_BLUE + C.WHITE + C.BOLD + f"  {titulo}".ljust(w) + C.RESET)
    if subtitulo:
        print(C.BG_BLUE + C.WHITE + f"  {subtitulo}".ljust(w) + C.RESET)
    print(C.BG_BLUE + C.WHITE + C.BOLD + " " * w + C.RESET)
    print()


def seccion(titulo):
    w = _w()
    print()
    print(C.CYAN + C.BOLD + "┌" + "─" * (w - 2) + "┐" + C.RESET)
    print(C.CYAN + C.BOLD + "│ " + titulo.ljust(w - 4) + " │" + C.RESET)
    print(C.CYAN + C.BOLD + "└" + "─" * (w - 2) + "┘" + C.RESET)


def ok(msg):    print(f"  {C.GREEN}[✓]{C.RESET} {msg}")
def info(msg):  print(f"  {C.BLUE}[i]{C.RESET} {msg}")
def warn(msg):  print(f"  {C.YELLOW}[!]{C.RESET} {msg}")
def err(msg):   print(f"  {C.RED}[✗]{C.RESET} {msg}")


def barra_progreso(actual, total, ancho=40, prefijo="", sufijo="", color=C.GREEN):
    if total <= 0:
        return
    frac = min(1.0, max(0.0, actual / total))
    llenos = int(ancho * frac)
    barra = "█" * llenos + C.DIM + "░" * (ancho - llenos) + C.RESET
    pct = f"{frac * 100:5.1f}%"
    sys.stdout.write(f"\r  {prefijo} {color}{barra}{C.RESET} {pct}  {sufijo}")
    sys.stdout.flush()
    if actual >= total:
        sys.stdout.write("\n")


def pedir(prompt, default=None, tipo=str, validador=None):
    sufijo = f" {C.DIM}[{default}]{C.RESET}" if default is not None else ""
    while True:
        try:
            raw = input(f"  {C.YELLOW}▶{C.RESET} {prompt}{sufijo}: ").strip()
        except (EOFError, KeyboardInterrupt):
            raise
        raw = raw.strip('"').strip("'")
        if not raw and default is not None:
            return default
        if not raw:
            warn("Valor obligatorio.")
            continue
        try:
            v = tipo(raw)
        except Exception:
            err(f"Valor inválido: '{raw}'. Se esperaba {tipo.__name__}.")
            continue
        if validador and not validador(v):
            err("Valor fuera de rango.")
            continue
        return v


def menu(titulo, opciones, permitir_salir=True):
    while True:
        seccion(titulo)
        for i, (clave, etiqueta, desc) in enumerate(opciones, 1):
            print(f"   {C.BOLD}{C.CYAN}{i}){C.RESET} "
                  f"{C.BOLD}{etiqueta}{C.RESET}  {C.DIM}{desc}{C.RESET}")
        if permitir_salir:
            print(f"   {C.BOLD}{C.RED}0){C.RESET} {C.BOLD}Salir{C.RESET}")
        print()
        try:
            sel = input(f"  {C.YELLOW}▶{C.RESET} Opción: ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if sel == "0" and permitir_salir:
            return None
        for i, (clave, _, _) in enumerate(opciones, 1):
            if sel == str(i):
                return clave
        err("Opción no válida.")


# ============================================================
# 1) LECTURA Y PREPARACIÓN
# ============================================================
def leer_archivo(ruta):
    if not os.path.isfile(ruta):
        raise FileNotFoundError(f"No existe: {ruta}")
    ext = ruta.rsplit(".", 1)[-1].lower()
    if ext == "csv":
        for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                return pd.read_csv(ruta, encoding=enc)
            except UnicodeDecodeError:
                continue
        return pd.read_csv(ruta)
    if ext in ("xlsx", "xls"):
        return pd.read_excel(ruta)
    raise ValueError(f"Formato no soportado: {ruta}")


def detectar_columna_id(df, target):
    for c in df.columns:
        if c == target:
            continue
        cl = c.lower()
        if "polig" in cl or cl in ("id", "id_poligono"):
            return c
    return None


def preparar(train, pred, target, id_col):
    comunes = [c for c in train.columns if c in pred.columns]
    if target not in train.columns:
        raise ValueError(f"'{target}' debe existir en TRAIN.")
    features = [c for c in comunes if c not in (target, id_col)]
    numericas = [
        c for c in features
        if pd.to_numeric(train[c], errors="coerce").notna().sum() > 0
    ]
    ignoradas = [c for c in features if c not in numericas]
    if ignoradas:
        warn(f"{len(ignoradas)} columnas ignoradas (no numéricas o no comunes)")
        for c in ignoradas[:5]:
            print(f"      {C.DIM}- {c}{C.RESET}")
        if len(ignoradas) > 5:
            print(f"      {C.DIM}  ... y {len(ignoradas) - 5} más{C.RESET}")

    for c in numericas:
        train[c] = pd.to_numeric(train[c], errors="coerce")
        pred[c] = pd.to_numeric(pred[c], errors="coerce")

    train = train.dropna(subset=[target]).reset_index(drop=True)
    y_train = train[target].astype(float)
    X_train = train[numericas].copy()
    X_pred = pred[numericas].copy()

    imp = SimpleImputer(strategy="median").fit(X_train)
    X_train = pd.DataFrame(imp.transform(X_train), columns=numericas, index=X_train.index)
    X_pred = pd.DataFrame(imp.transform(X_pred), columns=numericas, index=X_pred.index)

    ids_pred = (pred[id_col] if id_col and id_col in pred.columns
                else pd.Series(range(len(pred)), name="fila"))
    return X_train, y_train, X_pred, ids_pred, numericas, train


def limitar_k(k, n):
    return max(1, min(int(k), n))


# ============================================================
# 2) MODELO Y MÉTRICAS
# ============================================================
def nuevo_modelo(seed=42):
    return XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        objective="reg:squarederror", random_state=seed, n_jobs=-1,
    )


def calcular_metricas(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    r2 = r2_score(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    medae = float(np.median(np.abs(y_true - y_pred)))
    mask = np.abs(y_true) > 1e-8
    mape = (float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
            if mask.any() else np.nan)
    sesgo = float(np.mean(y_pred - y_true))
    if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        corr = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        corr = np.nan
    return {"R2": r2, "RMSE": rmse, "MAE": mae, "MedAE": medae,
            "MAPE": mape, "Sesgo": sesgo, "Corr": corr}


# ============================================================
# 3) K-FOLD MASIVO
# ============================================================
def correr_kfold_multi(X, y, cols, n_runs=1000, n_splits=5,
                       base_seed=1000, grupos=None):
    Xs = X[cols]
    registros_fold = []
    registros_run = []
    t0 = time.time()

    for run in range(1, n_runs + 1):
        seed = base_seed + run
        if grupos is not None:
            splitter = GroupKFold(n_splits=n_splits).split(Xs, y, groups=grupos)
        else:
            splitter = KFold(n_splits=n_splits, shuffle=True,
                             random_state=seed).split(Xs)

        r2s, rmses, maes, medaes, mapes, sesgos, corrs = ([] for _ in range(7))
        for fold_i, (tri, tei) in enumerate(splitter, 1):
            m = nuevo_modelo(seed=seed)
            m.fit(Xs.iloc[tri], y.iloc[tri])
            p = m.predict(Xs.iloc[tei])
            mm = calcular_metricas(y.iloc[tei].values, p)
            mm["Run"] = run
            mm["Fold"] = fold_i
            mm["N_test"] = int(len(tei))
            registros_fold.append(mm)
            r2s.append(mm["R2"]); rmses.append(mm["RMSE"]); maes.append(mm["MAE"])
            medaes.append(mm["MedAE"]); mapes.append(mm["MAPE"])
            sesgos.append(mm["Sesgo"]); corrs.append(mm["Corr"])

        registros_run.append({
            "Run": run,
            "R2_mean": np.nanmean(r2s),   "R2_std": np.nanstd(r2s),
            "RMSE_mean": np.nanmean(rmses), "RMSE_std": np.nanstd(rmses),
            "MAE_mean": np.nanmean(maes), "MAE_std": np.nanstd(maes),
            "MedAE_mean": np.nanmean(medaes),
            "MAPE_mean": np.nanmean(mapes), "MAPE_std": np.nanstd(mapes),
            "Sesgo_mean": np.nanmean(sesgos), "Sesgo_std": np.nanstd(sesgos),
            "Corr_mean": np.nanmean(corrs),
        })

        if run == 1 or run % max(1, n_runs // 200) == 0 or run == n_runs:
            el = time.time() - t0
            eta = (el / run) * (n_runs - run)
            sufijo = (f"R2={registros_run[-1]['R2_mean']:.3f}  "
                      f"RMSE={registros_run[-1]['RMSE_mean']:.3f}  "
                      f"ETA {int(eta)}s")
            barra_progreso(run, n_runs, sufijo=sufijo)

    return pd.DataFrame(registros_fold), pd.DataFrame(registros_run)


# ============================================================
# 4) RESUMEN ESTADÍSTICO
# ============================================================
def resumen_estadistico(df_runs, etiqueta):
    filas = []
    for b in ["R2", "RMSE", "MAE", "MedAE", "MAPE", "Sesgo", "Corr"]:
        col = b + "_mean"
        if col not in df_runs.columns:
            continue
        s = df_runs[col].dropna()
        n = len(s)
        if n == 0:
            continue
        media = float(s.mean())
        sd = float(s.std())
        ic = 1.96 * sd / np.sqrt(n) if n > 1 else np.nan
        filas.append({
            "Metodo": etiqueta, "Metrica": b, "N": n,
            "Media": media, "Mediana": float(s.median()),
            "Desv_std": sd, "Min": float(s.min()),
            "P05": float(s.quantile(0.05)), "P25": float(s.quantile(0.25)),
            "P75": float(s.quantile(0.75)), "P95": float(s.quantile(0.95)),
            "Max": float(s.max()),
            "IC95_inf": media - ic, "IC95_sup": media + ic,
        })
    return pd.DataFrame(filas)


def tabla_resumen(comparativo, metrica, orden="desc", color_mejor=C.GREEN):
    sub = comparativo[comparativo["Metrica"] == metrica].copy()
    if sub.empty:
        return
    sub = sub.sort_values("Media", ascending=(orden == "asc")).reset_index(drop=True)
    headers = ["#", "Método", "Vars", "Media", "Desv", "P05", "P95", "IC95%"]
    anchos = [3, 18, 5, 9, 8, 9, 9, 20]
    w = _w()
    if sum(anchos) + 8 > w:
        anchos[1] = max(10, w - (sum(anchos) - anchos[1]) - 8)

    def fila(vals, col=None, bold=False):
        partes = []
        for v, a in zip(vals, anchos):
            s = str(v)
            if len(s) > a:
                s = s[:a - 1] + "…"
            partes.append(s.ljust(a))
        linea = " │ ".join(partes)
        pref = C.BOLD if bold else ""
        col = col or ""
        print(f"  {pref}{col}{linea}{C.RESET}")

    fila(headers, col=C.CYAN, bold=True)
    print(f"  {C.DIM}{'─' * (sum(anchos) + 3 * (len(anchos) - 1))}{C.RESET}")

    for i, r in sub.iterrows():
        es_mejor = (i == 0)
        col = color_mejor if es_mejor else ""
        ic = f"[{r['IC95_inf']:.3f},{r['IC95_sup']:.3f}]"
        fila([
            i + 1, r["Metodo"], int(r["N_caracteristicas"]),
            f"{r['Media']:.4f}", f"{r['Desv_std']:.4f}",
            f"{r['P05']:.4f}", f"{r['P95']:.4f}", ic,
        ], col=col, bold=es_mejor)


# ============================================================
# 5) SELECCIÓN DE CARACTERÍSTICAS
# ============================================================
def preseleccionar_rapido(X, y, max_features=200, verbose=True):
    """Filtro previo: mantiene las top-`max_features` por MI + |corr|.
    Con p >> n esto acelera todo drásticamente y suele mejorar el resultado."""
    if X.shape[1] <= max_features:
        return X
    if verbose:
        info(f"Preseleccionando {max_features} de {X.shape[1]} features "
             f"(correlación + mutual information)...")
    corr = X.corrwith(y).abs().fillna(0)
    top_corr = corr.sort_values(ascending=False).head(max_features).index.tolist()

    # MI sobre un subconjunto más pequeño para no tardar
    pre_mi = corr.sort_values(ascending=False).head(min(400, X.shape[1])).index
    try:
        mi_vals = mutual_info_regression(X[pre_mi], y, random_state=42)
        mi = pd.Series(mi_vals, index=pre_mi).sort_values(ascending=False)
        top_mi = mi.head(max_features).index.tolist()
    except Exception:
        top_mi = []

    keep = list(dict.fromkeys(top_corr + top_mi))[:max_features]
    if verbose:
        ok(f"Preselección → {len(keep)} features")
    return X[keep]


def sel_correlacion(X, y, k):
    corr = X.corrwith(y).abs().sort_values(ascending=False)
    return list(corr.head(limitar_k(k, len(corr))).index)


def sel_mi(X, y, k):
    mi = pd.Series(mutual_info_regression(X, y, random_state=42), index=X.columns)
    return list(mi.sort_values(ascending=False).head(limitar_k(k, len(mi))).index)


def sel_kbest(X, y, k):
    k = limitar_k(k, X.shape[1])
    sel = SelectKBest(f_regression, k=k).fit(X, y)
    return list(X.columns[sel.get_support()])


def sel_lasso(X, y, k):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    corte = max(2, int(len(X) * 0.8))
    if len(X) - corte < 2:
        corte = max(1, len(X) // 2)
    Xa, Xb = Xs[:corte], Xs[corte:]
    ya, yb = y.values[:corte], y.values[corte:]
    mejor_alpha, mejor_error = None, np.inf
    for alpha in np.logspace(-4, 0, 40):
        m = Lasso(alpha=alpha, max_iter=20000).fit(Xa, ya)
        err = mean_squared_error(yb, m.predict(Xb))
        if err < mejor_error:
            mejor_error, mejor_alpha = err, alpha
    m = Lasso(alpha=mejor_alpha, max_iter=20000).fit(Xs, y)
    coef = pd.Series(np.abs(m.coef_), index=X.columns)
    cols = list(coef[coef > 1e-8].sort_values(ascending=False).index)
    if not cols:
        cols = list(coef.sort_values(ascending=False).head(limitar_k(k, len(coef))).index)
    return cols[:limitar_k(k, len(cols))]


def sel_rf(X, y, k):
    m = RandomForestRegressor(300, random_state=42, n_jobs=-1,
                              min_samples_leaf=2).fit(X, y)
    imp = pd.Series(m.feature_importances_, index=X.columns).sort_values(ascending=False)
    return list(imp.head(limitar_k(k, len(imp))).index)


def sel_xgb(X, y, k):
    m = nuevo_modelo(seed=42).fit(X, y)
    imp = pd.Series(m.feature_importances_, index=X.columns).sort_values(ascending=False)
    return list(imp.head(limitar_k(k, len(imp))).index)


# ============================================================
# 6) FLUJO PRINCIPAL
# ============================================================
def ejecutar(train_path, pred_path, target, id_col, k, folds, n_runs,
             salida_dir, metodos=("LASSO", "RF", "XGB"),
             usar_group=False, incluir_baseline=False, runs_baseline=30,
             preseleccionar=0):

    banner("ANÁLISIS DE ROBUSTEZ · K-FOLD CV MASIVO",
           f"{n_runs} corridas × {folds} folds · XGBoost + selección de características")
    t_ini = time.time()

    # --- 1. Carga ---
    seccion("1 / 6   CARGANDO DATOS")
    train = leer_archivo(train_path); ok(f"TRAIN: {train_path}")
    pred = leer_archivo(pred_path);   ok(f"PRED : {pred_path}")

    if id_col is None:
        id_col = detectar_columna_id(train, target)
        if id_col:
            info(f"Columna ID detectada: '{id_col}'")

    X_train, y_train, X_pred, ids_pred, features, train_df = preparar(
        train, pred, target, id_col)
    k = limitar_k(k, len(features))

    print()
    print(f"   {C.BOLD}Resumen de datos:{C.RESET}")
    print(f"     · Filas train      : {C.CYAN}{len(X_train)}{C.RESET}")
    print(f"     · Filas predicción : {C.CYAN}{len(X_pred)}{C.RESET}")
    print(f"     · Features         : {C.CYAN}{len(features)}{C.RESET}")
    print(f"     · Target           : {C.CYAN}{target}{C.RESET}")
    print(f"     · ID               : {C.CYAN}{id_col}{C.RESET}")
    print(f"     · K                : {C.CYAN}{k}{C.RESET}")
    print(f"     · Corridas         : {C.CYAN}{n_runs}{C.RESET}")
    print(f"     · Folds            : {C.CYAN}{folds}{C.RESET}")
    print(f"     · Métodos          : {C.CYAN}{', '.join(metodos)}{C.RESET}")
    if preseleccionar > 0 and len(features) > preseleccionar:
        print(f"     · Preselección     : {C.CYAN}{preseleccionar} features{C.RESET}")
    if incluir_baseline:
        print(f"     · Baseline TODAS   : {C.YELLOW}incluido "
              f"(solo {runs_baseline} corridas, no compite){C.RESET}")

    os.makedirs(salida_dir, exist_ok=True)

    grupos = None
    if usar_group and id_col and id_col in train_df.columns:
        grupos = train_df[id_col].values
        info(f"GroupKFold activado · {len(np.unique(grupos))} grupos únicos")

    # --- 2. Preselección opcional ---
    X_base = X_train
    features_base = features
    if preseleccionar > 0 and len(features) > preseleccionar:
        seccion("2 / 6   PRESELECCIÓN DE CARACTERÍSTICAS")
        X_base = preseleccionar_rapido(X_train, y_train, max_features=preseleccionar)
        features_base = list(X_base.columns)

    # --- 3. Selección fina ---
    seccion("3 / 6   SELECCIÓN DE CARACTERÍSTICAS")
    conjuntos = {}

    for met in metodos:
        met_u = met.upper()
        try:
            t0 = time.time()
            if met_u in ("CORRELACION", "CORR"):
                conjuntos["Correlacion"] = sel_correlacion(X_base, y_train, k)
            elif met_u in ("MI", "MUTUAL_INFO"):
                conjuntos["MutualInfo"] = sel_mi(X_base, y_train, k)
            elif met_u in ("KBEST", "SELECTKBEST"):
                conjuntos["SelectKBest"] = sel_kbest(X_base, y_train, k)
            elif met_u == "LASSO":
                conjuntos["LASSO"] = sel_lasso(X_base, y_train, k)
            elif met_u in ("RF", "RANDOM_FOREST"):
                conjuntos["RandomForest"] = sel_rf(X_base, y_train, k)
            elif met_u in ("XGB", "XGBOOST"):
                conjuntos["XGBoost"] = sel_xgb(X_base, y_train, k)
            elif met_u in ("TODAS", "BASELINE", "ALL"):
                if incluir_baseline:
                    conjuntos["TODAS"] = features
                    warn(f"{met}: baseline incluido SOLO como referencia "
                         f"({runs_baseline} corridas)")
                else:
                    info(f"{met}: omitido (usa --incluir-baseline para forzarlo)")
                continue
            else:
                warn(f"Método desconocido: {met}")
                continue

            last = list(conjuntos)[-1]
            ok(f"{met:20s} → {len(conjuntos[last]):4d} vars  "
               f"({time.time() - t0:.2f}s)")
        except Exception as e:
            err(f"{met}: {e}")

    if not conjuntos:
        err("No hay conjuntos de características para evaluar.")
        return

    # --- 4. Corridas masivas ---
    seccion(f"4 / 6   {n_runs} CORRIDAS × {folds}-FOLD CV POR MÉTODO")
    resumenes = []
    for idx, (nombre, cols) in enumerate(conjuntos.items(), 1):
        cols = [c for c in cols if c in X_train.columns]
        if not cols:
            continue

        es_baseline = (nombre == "TODAS")
        n_runs_ef = runs_baseline if es_baseline else n_runs

        print()
        etiqueta_extra = (f" {C.YELLOW}(baseline, {n_runs_ef} corridas, no compite)"
                          f"{C.RESET}" if es_baseline else "")
        print(f"  {C.BOLD}{C.MAGENTA}[{idx}/{len(conjuntos)}]{C.RESET} "
              f"{C.BOLD}{nombre}{C.RESET}  "
              f"{C.DIM}({len(cols)} variables · {n_runs_ef} corridas){C.RESET}"
              f"{etiqueta_extra}")

        t0 = time.time()
        df_folds, df_runs = correr_kfold_multi(
            X_train, y_train, cols,
            n_runs=n_runs_ef, n_splits=folds,
            base_seed=1000, grupos=grupos)

        df_folds.to_csv(os.path.join(salida_dir, f"folds_1000_{nombre}.csv"), index=False)
        df_runs.to_csv(os.path.join(salida_dir, f"runs_1000_{nombre}.csv"), index=False)
        pd.DataFrame({"caracteristica": cols}).to_csv(
            os.path.join(salida_dir, f"caracteristicas_{nombre}.csv"), index=False)

        res = resumen_estadistico(df_runs, nombre)
        res["N_caracteristicas"] = len(cols)
        resumenes.append(res)

        ok(f"Terminado {nombre} en {time.time() - t0:.1f}s")

    # --- 5. Comparativo ---
    seccion("5 / 6   COMPARATIVO ENTRE MÉTODOS")
    comparativo = pd.concat(resumenes, ignore_index=True)
    comparativo.to_csv(os.path.join(salida_dir, "comparativo_completo.csv"), index=False)

    for m in ["R2", "RMSE", "MAE", "MAPE", "Corr"]:
        sub = comparativo[comparativo["Metrica"] == m]
        if sub.empty:
            continue
        sub = sub[["Metodo", "N_caracteristicas", "Media", "Desv_std",
                   "Mediana", "P05", "P95", "IC95_inf", "IC95_sup"]]
        sub.to_csv(os.path.join(salida_dir, f"comparativo_{m}.csv"), index=False)

    for m, orden in [("R2", "desc"), ("RMSE", "asc"), ("MAE", "asc"),
                     ("MAPE", "asc"), ("Corr", "desc")]:
        print()
        print(f"  {C.BOLD}{C.WHITE}▶ Métrica: {m}   "
              f"{C.DIM}({'mayor es mejor' if orden == 'desc' else 'menor es mejor'})"
              f"{C.RESET}")
        tabla_resumen(comparativo, m, orden=orden)

    # --- 6. Modelo final ---
    seccion("6 / 6   MODELO FINAL")
    r2s = comparativo[(comparativo["Metrica"] == "R2") &
                      (comparativo["Metodo"] != "TODAS")] \
        .sort_values("Media", ascending=False)
    if r2s.empty:
        err("No hay métodos válidos (solo baseline).")
        return

    mejor_metodo = r2s.iloc[0]["Metodo"]
    cols_finales = [c for c in conjuntos[mejor_metodo] if c in X_train.columns]
    ok(f"Mejor método (R² medio): {C.BOLD}{mejor_metodo}{C.RESET} "
       f"({len(cols_finales)} características)")

    print(f"  {C.DIM}Entrenando XGBoost con TODO train...{C.RESET}")
    m_final = nuevo_modelo(seed=42).fit(X_train[cols_finales], y_train)
    pred_final = m_final.predict(X_pred[cols_finales])

    out = pred.copy()
    out[target + "_PREDICHO"] = pred_final
    ruta_pred = os.path.join(salida_dir, "predicciones_finales.csv")
    out.to_csv(ruta_pred, index=False)
    ok(f"Predicciones guardadas en: {ruta_pred}")

    with open(os.path.join(salida_dir, "info_ejecucion.txt"), "w", encoding="utf-8") as f:
        f.write(f"Fecha: {pd.Timestamp.now()}\n")
        f.write(f"Train: {train_path}\nPrediccion: {pred_path}\n")
        f.write(f"Target: {target}\nID: {id_col}\nK: {k}\n")
        f.write(f"Folds: {folds}\nCorridas: {n_runs}\n")
        f.write(f"Metodos: {list(metodos)}\nGroupKFold: {usar_group}\n")
        f.write(f"Baseline incluido: {incluir_baseline}\n")
        f.write(f"Preseleccion: {preseleccionar}\n")
        f.write(f"Mejor metodo: {mejor_metodo}\nN features final: {len(cols_finales)}\n")

    print()
    print(f"  {C.GREEN}{C.BOLD}✓ Proceso completado{C.RESET}")
    print(f"     · Tiempo total : {time.time() - t_ini:.1f}s")
    print(f"     · Carpeta      : {C.CYAN}{os.path.abspath(salida_dir)}{C.RESET}")
    print()


# ============================================================
# 7) MENÚ INTERACTIVO
# ============================================================
def menu_principal():
    while True:
        clear()
        banner("XGBoost · 1000 CORRIDAS DE K-FOLD CV",
               "Análisis de robustez y variabilidad con selección de características")
        opcion = menu("MENÚ PRINCIPAL", [
            ("correr", "▶  Ejecutar análisis",
             "configura y lanza las 1000 corridas"),
            ("demo",   "⚡  Demo rápida",
             "50 corridas, 5-fold, LASSO+RF+XGB (para probar)"),
            ("ver",    "📂  Ver resultados anteriores",
             "leer un comparativo_completo.csv"),
            ("salir",  "✗  Salir", ""),
        ], permitir_salir=False)

        if opcion in (None, "salir"):
            print(f"\n  {C.CYAN}Hasta luego.{C.RESET}\n")
            return
        try:
            if opcion == "correr":
                configurar_y_correr()
            elif opcion == "demo":
                demo()
            elif opcion == "ver":
                ver_resultados()
        except (KeyboardInterrupt, EOFError):
            print(f"\n  {C.YELLOW}Interrumpido.{C.RESET}")

        input(f"\n  {C.DIM}Presiona Enter para volver al menú...{C.RESET}")


def configurar_y_correr():
    clear()
    banner("CONFIGURACIÓN")
    train_path = pedir("Ruta TRAIN (.csv/.xlsx)", tipo=str)
    pred_path = pedir("Ruta PREDICCION (.csv/.xlsx)", tipo=str)
    target = pedir("Variable OBJETIVO (ej. RENDIMIENTO_T_HA)", tipo=str)

    try:
        cols = leer_archivo(train_path).columns.tolist()
        if target not in cols:
            err(f"'{target}' no está en TRAIN. Columnas disponibles:")
            for i, c in enumerate(cols):
                print(f"      {C.DIM}{i+1:3d}. {c}{C.RESET}")
            return
        n_features_estimado = len([c for c in cols if c not in (target,)])
    except Exception as e:
        err(f"No se pudo leer TRAIN: {e}")
        return

    id_col = pedir("Columna ID (Enter = detectar)", default="", tipo=str) or None
    k = pedir("K de características", default=15, tipo=int,
              validador=lambda v: v > 0)
    folds = pedir("Folds del k-fold", default=5, tipo=int,
                  validador=lambda v: 2 <= v <= 20)
    n_runs = pedir("Número de corridas", default=1000, tipo=int,
                   validador=lambda v: v > 0)

    print()
    info("Métodos disponibles: LASSO, RF, XGB, MI, KBEST, CORRELACION")
    info("Por defecto NO se incluye 'TODAS' (baseline) — es el que más tarda.")
    met_txt = pedir("Métodos separados por coma",
                    default="LASSO,RF,XGB,MI,KBEST,CORRELACION", tipo=str)
    metodos = [m.strip() for m in met_txt.split(",") if m.strip()]

    incluir_baseline = False
    runs_baseline = 30
    if n_features_estimado > 200:
        print()
        warn(f"Se detectaron {n_features_estimado} columnas. Recomendado preseleccionar.")
        preseleccionar = pedir(
            "Preseleccionar top-N features (0 = sin preselección)",
            default=200, tipo=int, validador=lambda v: v >= 0)
    else:
        preseleccionar = 0

    if preseleccionar == 0:
        incluir_baseline = pedir(
            "¿Incluir 'TODAS' como baseline? (s/N)",
            default="n", tipo=str).lower().startswith("s")
        if incluir_baseline:
            runs_baseline = pedir("Corridas para el baseline (recomendado 30)",
                                  default=30, tipo=int, validador=lambda v: v > 0)

    usar_group = pedir("¿Usar GroupKFold por ID? (s/N)",
                       default="n", tipo=str).lower().startswith("s")
    salida = pedir("Carpeta de salida", default="./resultado_1000", tipo=str)

    print()
    print(f"  {C.BOLD}Resumen de la configuración:{C.RESET}")
    print(f"     · TRAIN          : {C.CYAN}{train_path}{C.RESET}")
    print(f"     · PRED           : {C.CYAN}{pred_path}{C.RESET}")
    print(f"     · Target         : {C.CYAN}{target}{C.RESET}")
    print(f"     · ID             : {C.CYAN}{id_col}{C.RESET}")
    print(f"     · K / Folds      : {C.CYAN}{k} / {folds}{C.RESET}")
    print(f"     · Corridas       : {C.CYAN}{n_runs}{C.RESET}")
    print(f"     · Métodos        : {C.CYAN}{', '.join(metodos)}{C.RESET}")
    print(f"     · Preselección   : {C.CYAN}{preseleccionar or 'no'}{C.RESET}")
    print(f"     · Baseline TODAS : {C.CYAN}{incluir_baseline}{C.RESET}")
    print(f"     · GroupKFold     : {C.CYAN}{usar_group}{C.RESET}")
    print(f"     · Salida         : {C.CYAN}{salida}{C.RESET}")
    print()
    confirmar = input(f"  {C.YELLOW}▶{C.RESET} ¿Comenzar? (s/N): ").strip().lower()
    if not confirmar.startswith("s"):
        warn("Cancelado.")
        return

    ejecutar(train_path, pred_path, target, id_col, k, folds, n_runs,
             salida, metodos=metodos, usar_group=usar_group,
             incluir_baseline=incluir_baseline, runs_baseline=runs_baseline,
             preseleccionar=preseleccionar)


def demo():
    clear()
    banner("DEMO RÁPIDA", "50 corridas × 5 folds · LASSO + RF + XGB")
    train_path = pedir("Ruta TRAIN", tipo=str)
    pred_path = pedir("Ruta PREDICCION", tipo=str)
    target = pedir("Variable OBJETIVO", tipo=str)
    salida = pedir("Carpeta de salida", default="./demo_1000", tipo=str)
    ejecutar(train_path, pred_path, target, None, 15, 5, 50,
             salida, metodos=["LASSO", "RF", "XGB"],
             usar_group=False, incluir_baseline=False, preseleccionar=200)


def ver_resultados():
    clear()
    banner("VER RESULTADOS ANTERIORES")
    carpeta = pedir("Carpeta con comparativo_completo.csv",
                    default="./resultado_1000", tipo=str)
    ruta = os.path.join(carpeta, "comparativo_completo.csv")
    if not os.path.isfile(ruta):
        err(f"No existe: {ruta}")
        return
    comp = pd.read_csv(ruta)
    ok(f"Cargado: {ruta}  ({len(comp)} filas)")

    for m, orden in [("R2", "desc"), ("RMSE", "asc"), ("MAE", "asc"),
                     ("MAPE", "asc"), ("Corr", "desc")]:
        print()
        print(f"  {C.BOLD}{C.WHITE}▶ Métrica: {m}   "
              f"{C.DIM}({'mayor es mejor' if orden == 'desc' else 'menor es mejor'})"
              f"{C.RESET}")
        tabla_resumen(comp, m, orden=orden)

    mejor = comp[(comp["Metrica"] == "R2") & (comp["Metodo"] != "TODAS")] \
        .sort_values("Media", ascending=False)
    if not mejor.empty:
        print()
        ok(f"Mejor método: {C.BOLD}{mejor.iloc[0]['Metodo']}{C.RESET}  "
           f"(R² = {mejor.iloc[0]['Media']:.4f} ± {mejor.iloc[0]['Desv_std']:.4f})")


# ============================================================
# 8) CLI
# ============================================================
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train")
    ap.add_argument("--prediccion")
    ap.add_argument("--target")
    ap.add_argument("--id", default=None)
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--runs", type=int, default=1000)
    ap.add_argument("--metodos", default="LASSO,RF,XGB,MI,KBEST,CORRELACION",
                    help="Métodos separados por coma. Por defecto NO incluye TODAS.")
    ap.add_argument("--group", action="store_true",
                    help="Usar GroupKFold por la columna ID")
    ap.add_argument("--incluir-baseline", action="store_true",
                    help="Añadir 'TODAS' como baseline con pocas corridas. NO recomendado con muchas features.")
    ap.add_argument("--runs-baseline", type=int, default=30,
                    help="Número de corridas para el baseline (default: 30)")
    ap.add_argument("--preseleccionar", type=int, default=0,
                    help="Preseleccionar top-N features antes de la selección fina (0 = desactivado)")
    ap.add_argument("--salida", default="./resultado_1000")
    ap.add_argument("--menu", action="store_true",
                    help="Forzar el menú interactivo")
    args = ap.parse_args()

    if args.menu or not (args.train and args.prediccion and args.target):
        try:
            menu_principal()
        except (KeyboardInterrupt, EOFError):
            print("\n  Interrumpido por el usuario.")
    else:
        metodos = [m.strip() for m in args.metodos.split(",") if m.strip()]
        ejecutar(args.train, args.prediccion, args.target, args.id,
                 args.k, args.folds, args.runs, args.salida,
                 metodos=metodos, usar_group=args.group,
                 incluir_baseline=args.incluir_baseline,
                 runs_baseline=args.runs_baseline,
                 preseleccionar=args.preseleccionar)


if __name__ == "__main__":
    main()