#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SELECTOR DE CARACTERÍSTICAS + XGBOOST (train / predicción, con k-fold CV)

Recibe DOS archivos (.csv o .xlsx):
  1) TRAIN      -> tiene la variable objetivo. Se usa para TODO: seleccionar
                   características, evaluarlas (con validación cruzada,
                   no con un test fijo) y entrenar el modelo final.
  2) PREDICCION -> NO tiene la variable objetivo (o la tiene vacía);
                   son las parcelas/filas para las que quieres el resultado final

Por qué k-fold y no un test fijo: con pocas filas de entrenamiento, apartar un
test fijo desperdicia datos y el resultado depende mucho de qué filas cayeron
ahí por azar. En vez de eso, cada método de selección se evalúa con
RepeatedKFold sobre TODO train (varias particiones, se promedia el resultado),
que es más estable.

Para cada método de selección de características (Correlación, Mutual
Information, SelectKBest, RFE, LASSO, Random Forest, XGBoost, y opcionalmente
Sequential Feature Selection) hace:
  - selecciona usando TRAIN completo
  - evalúa esas características con XGBoost por validación cruzada (R2, RMSE, MAE)

Al final:
  - guarda un resumen comparando todos los métodos + el baseline (todas las características)
  - se queda con el MEJOR método según la validación cruzada
  - entrena XGBoost con TODO train (esas características)
  - genera las predicciones finales para el archivo de PREDICCION

Requiere: pip install xgboost   (si no lo tienes, te avisa y no continúa)

USO
---
Modo con menú (te va preguntando):
    python3 xgb_seleccion_prediccion.py

Modo directo (sin menú):
    python3 xgb_seleccion_prediccion.py --train train.xlsx \
        --prediccion prediccion.xlsx --target RENDIMIENTO_T_HA --id ID_POLIGONO --k 15
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

from sklearn.feature_selection import (
    VarianceThreshold,
    mutual_info_regression,
    SelectKBest,
    f_regression,
    RFE,
    SequentialFeatureSelector,
)
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor
except ImportError:
    sys.exit(
        "\nEste script necesita la librería 'xgboost', que no está instalada.\n"
        "Instálala con:  pip install xgboost\n"
    )


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
    elif ext in ("xlsx", "xls"):
        return pd.read_excel(ruta)
    raise ValueError(f"Formato no soportado: {ruta} (usa .csv, .xlsx o .xls)")


def detectar_columna_id(df, target):
    for c in df.columns:
        if c == target:
            continue
        cl = c.lower()
        if "polig" in cl or cl in ("id", "id_poligono"):
            return c
    return None


def preparar(train, pred, target, id_col):
    """Se queda con las columnas numéricas comunes a TRAIN y PREDICCION
    (menos el ID y el target), imputa y devuelve todo listo."""
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
        print("Columnas ignoradas (no numéricas o no comunes a train/predicción):")
        for c in ignoradas:
            print("  -", c)

    for c in numericas:
        train[c] = pd.to_numeric(train[c], errors="coerce")
        pred[c] = pd.to_numeric(pred[c], errors="coerce")

    train = train.dropna(subset=[target]).reset_index(drop=True)

    y_train = train[target].astype(float)
    X_train, X_pred = train[numericas].copy(), pred[numericas].copy()

    imp = SimpleImputer(strategy="median").fit(X_train)
    X_train = pd.DataFrame(imp.transform(X_train), columns=numericas, index=X_train.index)
    X_pred = pd.DataFrame(imp.transform(X_pred), columns=numericas, index=X_pred.index)

    ids_pred = pred[id_col] if id_col and id_col in pred.columns else pd.Series(range(len(pred)), name="fila")
    return X_train, y_train, X_pred, ids_pred, numericas


def limitar_k(k, n):
    return max(1, min(int(k), n))


# ============================================================
# 2) MODELO Y EVALUACIÓN (siempre XGBoost, con k-fold CV)
# ============================================================
def nuevo_modelo():
    return XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, objective="reg:squarederror",
        random_state=42, n_jobs=-1,
    )


def evaluar_cv(X, y, n_splits=5, n_repeats=5):
    """Evalúa por validación cruzada repetida (más estable que un test fijo
    cuando hay pocas filas). Devuelve el promedio de R2, RMSE y MAE."""
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    r2s, rmses, maes = [], [], []
    for tri, tei in cv.split(X):
        m = nuevo_modelo().fit(X.iloc[tri], y.iloc[tri])
        p = m.predict(X.iloc[tei])
        r2s.append(r2_score(y.iloc[tei], p))
        rmses.append(mean_squared_error(y.iloc[tei], p) ** 0.5)
        maes.append(mean_absolute_error(y.iloc[tei], p))
    return {"R2": np.mean(r2s), "R2_sd": np.std(r2s), "RMSE": np.mean(rmses), "MAE": np.mean(maes)}


# ============================================================
# 3) MÉTODOS DE SELECCIÓN (ajustados SOLO con TRAIN)
# ============================================================
def preseleccionar_rapido(X, y, max_features=200):
    """Para métodos lentos (RFE, SFS): si hay demasiadas columnas, primero se
    queda con las 'max_features' de mayor correlación con el objetivo. Esto
    evita que RFE/SFS tengan que evaluar >1000 columnas una por una."""
    if X.shape[1] <= max_features:
        return X
    top = X.corrwith(y).abs().sort_values(ascending=False).head(max_features).index
    return X[top]


def sel_variance(X):
    sel = VarianceThreshold(0.0).fit(X)
    cols = list(X.columns[sel.get_support()])
    return cols or list(X.columns)


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


def sel_rfe(X, y, k):
    k = limitar_k(k, X.shape[1])
    Xp = preseleccionar_rapido(X, y, max_features=200)
    if Xp.shape[1] < X.shape[1]:
        print(f"    (RFE: preseleccionando las {Xp.shape[1]} de mayor correlación antes de RFE, "
              f"de {X.shape[1]} originales, para que no tarde una eternidad)")
    est = RandomForestRegressor(150, random_state=42, n_jobs=-1, min_samples_leaf=2)
    sel = RFE(est, n_features_to_select=k, step=0.1).fit(Xp, y)
    return list(Xp.columns[sel.support_])


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
    return cols[:limitar_k(k, len(cols))], mejor_alpha


def sel_rf(X, y, k):
    m = RandomForestRegressor(300, random_state=42, n_jobs=-1, min_samples_leaf=2).fit(X, y)
    imp = pd.Series(m.feature_importances_, index=X.columns).sort_values(ascending=False)
    return list(imp.head(limitar_k(k, len(imp))).index)


def sel_xgb(X, y, k):
    m = nuevo_modelo().fit(X, y)
    imp = pd.Series(m.feature_importances_, index=X.columns).sort_values(ascending=False)
    return list(imp.head(limitar_k(k, len(imp))).index)


def sel_sfs(X, y, k):
    k = limitar_k(k, X.shape[1])
    Xp = preseleccionar_rapido(X, y, max_features=60)
    if Xp.shape[1] < X.shape[1]:
        print(f"    (SFS: preseleccionando las {Xp.shape[1]} de mayor correlación antes de SFS, "
              f"de {X.shape[1]} originales; SFS evalúa cada columna una por una y con miles "
              f"de candidatas sería impráctico)")
    est = RandomForestRegressor(100, random_state=42, n_jobs=-1, min_samples_leaf=2)
    sel = SequentialFeatureSelector(est, n_features_to_select=k, direction="forward",
                                     scoring="r2", cv=3, n_jobs=-1).fit(Xp, y)
    return list(Xp.columns[sel.get_support()])


# ============================================================
# 4) FLUJO PRINCIPAL
# ============================================================
def ejecutar(train_path, pred_path, target, id_col, k, folds, repeats, correr_sfs, salida_dir):
    train = leer_archivo(train_path)
    pred = leer_archivo(pred_path)

    if id_col is None:
        id_col = detectar_columna_id(train, target)
        if id_col:
            print(f"Columna de ID detectada automáticamente: '{id_col}'")

    X_train, y_train, X_pred, ids_pred, features = preparar(train, pred, target, id_col)
    k = limitar_k(k, len(features))
    print(f"\nCaracterísticas numéricas comunes a train/predicción: {len(features)}")
    print(f"Filas -> train: {len(X_train)}  predicción: {len(X_pred)}")
    print(f"Validación cruzada: {folds} folds x {repeats} repeticiones")
    print(f"K para métodos que lo requieren: {k}")
    if len(features) > 300:
        print(f"AVISO: {len(features)} características para {len(X_train)} filas de train es "
              f"MUCHO (más columnas que filas x8). Correlación/MI/SelectKBest/LASSO/RF/XGBoost "
              f"corren igual de rápido, pero RFE y SFS preseleccionan primero un subconjunto "
              f"por correlación para no tardar horas.")
    print()

    os.makedirs(salida_dir, exist_ok=True)

    resultados = []
    conjuntos = {"BASELINE_TODAS": features}

    print("Evaluando BASELINE (todas las características) con XGBoost + CV...")
    m = evaluar_cv(X_train[features], y_train, folds, repeats)
    resultados.append({"Metodo": "BASELINE_TODAS", "N_caracteristicas": len(features), **m})

    pasos = [
        ("Variance_Threshold", lambda: sel_variance(X_train)),
        ("Correlacion", lambda: sel_correlacion(X_train, y_train, k)),
        ("Mutual_Information", lambda: sel_mi(X_train, y_train, k)),
        ("SelectKBest", lambda: sel_kbest(X_train, y_train, k)),
        ("RFE", lambda: sel_rfe(X_train, y_train, k)),
        ("Random_Forest", lambda: sel_rf(X_train, y_train, k)),
        ("XGBoost_Importance", lambda: sel_xgb(X_train, y_train, k)),
    ]
    if correr_sfs:
        pasos.append(("Sequential_Feature_Selection", lambda: sel_sfs(X_train, y_train, k)))

    for i, (nombre, fn) in enumerate(pasos, 1):
        print(f"[{i}/{len(pasos)}] {nombre}...")
        try:
            conjuntos[nombre] = fn()
        except Exception as e:
            print("  ERROR:", e)

    print(f"[{len(pasos)+1}/{len(pasos)+1}] LASSO...")
    try:
        cols, alpha = sel_lasso(X_train, y_train, k)
        conjuntos["LASSO"] = cols
        print(f"  alpha elegido: {alpha:.6f}")
    except Exception as e:
        print("  ERROR:", e)

    print("\n" + "=" * 70)
    print("EVALUANDO CADA CONJUNTO DE CARACTERÍSTICAS CON VALIDACIÓN CRUZADA (XGBoost)")
    print("=" * 70)
    for nombre, cols in conjuntos.items():
        if nombre == "BASELINE_TODAS" or not cols:
            continue
        cols = [c for c in cols if c in X_train.columns]
        m = evaluar_cv(X_train[cols], y_train, folds, repeats)
        resultados.append({"Metodo": nombre, "N_caracteristicas": len(cols), **m})
        pd.DataFrame({"caracteristica": cols}).to_csv(
            os.path.join(salida_dir, f"caracteristicas_{nombre}.csv"), index=False
        )
        print(f"{nombre:32s} | {len(cols):3d} vars | R2={m['R2']:.4f} (+-{m['R2_sd']:.3f}) | "
              f"RMSE={m['RMSE']:.4f} | MAE={m['MAE']:.4f}")

    resumen = pd.DataFrame(resultados).sort_values("R2", ascending=False).reset_index(drop=True)
    resumen.to_csv(os.path.join(salida_dir, "resumen_resultados.csv"), index=False)
    print("\n" + "=" * 70)
    print("RESUMEN (ordenado por R2 promedio en validación cruzada, mayor es mejor)")
    print("=" * 70)
    print(resumen.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # -------- mejor método -> predicciones finales --------
    mejor = resumen.iloc[0]["Metodo"]
    cols_finales = conjuntos[mejor]
    cols_finales = [c for c in cols_finales if c in X_train.columns]
    print(f"\nMejor método según validación cruzada: {mejor}  ({len(cols_finales)} características)")
    print("Entrenando XGBoost con TODO train para la predicción final...")

    modelo_final = nuevo_modelo().fit(X_train[cols_finales], y_train)
    pred_valores = modelo_final.predict(X_pred[cols_finales])

    salida_pred = pred.copy()
    salida_pred[target + "_PREDICHO"] = pred_valores
    ruta_pred = os.path.join(salida_dir, "predicciones_finales.csv")
    salida_pred.to_csv(ruta_pred, index=False)

    print(f"\nPredicciones guardadas en: {ruta_pred}")
    print(f"Todo lo demás (resumen, listas de características) en: {os.path.abspath(salida_dir)}")


# ============================================================
# 5) INTERFAZ / ARGUMENTOS
# ============================================================
def menu_interactivo():
    print("=" * 70)
    print("  SELECCIÓN DE CARACTERÍSTICAS + XGBOOST (train / predicción, k-fold CV)")
    print("=" * 70)
    train_path = input("Ruta de TRAIN (.csv/.xlsx): ").strip().strip('"').strip("'")
    pred_path = input("Ruta de PREDICCION (.csv/.xlsx, sin target): ").strip().strip('"').strip("'")
    target = input("Nombre de la variable OBJETIVO (ej. RENDIMIENTO_T_HA): ").strip()
    id_txt = input("Columna de ID de la parcela [Enter = detectar sola]: ").strip()
    id_col = id_txt or None
    k_txt = input("¿Cuántas características conservar en los métodos con K? [Enter = 15]: ").strip()
    k = int(k_txt) if k_txt else 15
    folds_txt = input("¿Folds de la validación cruzada? [Enter = 5]: ").strip()
    folds = int(folds_txt) if folds_txt else 5
    rep_txt = input("¿Repeticiones de la validación cruzada? [Enter = 5]: ").strip()
    repeats = int(rep_txt) if rep_txt else 5
    sfs = input("¿Correr también Sequential Feature Selection? (tarda) [s/N]: ").strip().lower().startswith("s")
    salida = input("Carpeta de salida [Enter = ./resultado_xgb]: ").strip() or "./resultado_xgb"
    ejecutar(train_path, pred_path, target, id_col, k, folds, repeats, sfs, salida)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train")
    ap.add_argument("--prediccion")
    ap.add_argument("--target")
    ap.add_argument("--id", default=None, help="Columna de ID (si no se indica, se detecta sola)")
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--sfs", action="store_true", help="Correr también Sequential Feature Selection")
    ap.add_argument("--salida", default="./resultado_xgb")
    args = ap.parse_args()

    if args.train and args.prediccion and args.target:
        ejecutar(args.train, args.prediccion, args.target, args.id, args.k,
                  args.folds, args.repeats, args.sfs, args.salida)
    else:
        try:
            menu_interactivo()
        except (KeyboardInterrupt, EOFError):
            print("\nInterrumpido por el usuario.")


if __name__ == "__main__":
    main()