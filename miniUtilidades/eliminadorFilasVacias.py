import os
import pandas as pd


def obtener_archivo():
    print("\n========================================")
    print("       FILTRADOR DE DATOS")
    print("========================================")
    print("1. CSV")
    print("2. Excel")
    print("========================================")

    ruta = input("\nIntroduce la ruta del archivo: ").strip().strip('"')

    if not os.path.isfile(ruta):
        print("\n❌ El archivo no existe.")
        return None

    return ruta


def filtrar_archivo(ruta):
    extension = os.path.splitext(ruta)[1].lower()

    try:
        # Leer archivo
        if extension == ".csv":
            df = pd.read_csv(ruta)
        elif extension in [".xlsx", ".xls"]:
            df = pd.read_excel(ruta)
        else:
            print("\n❌ Formato no compatible.")
            print("Usa .csv, .xlsx o .xls")
            return

    except Exception as e:
        print(f"\n❌ No se pudo abrir el archivo:\n{e}")
        return

    print("\n----------------------------------------")
    print(f"Archivo: {os.path.basename(ruta)}")
    print(f"Filas encontradas: {len(df)}")
    print(f"Columnas encontradas: {len(df.columns)}")
    print("----------------------------------------")

    if len(df.columns) < 4:
        print("\n❌ El archivo tiene menos de 4 columnas.")
        return

    # -------------------------------------------------
    # Columnas que NO se utilizarán para el filtro:
    # A, B, C y CR
    # -------------------------------------------------

    # Índices:
    # A  = 0
    # B  = 1
    # C  = 2
    # CR = 95
    #
    # Por lo tanto usamos todas las columnas excepto
    # A, B, C y CR.

    columnas_ignorar = {0, 1, 2, 95}

    columnas_filtro = [
        i for i in range(len(df.columns))
        if i not in columnas_ignorar
    ]

    # Evitar error si CR no existe
    columnas_filtro = [
        i for i in columnas_filtro
        if i < len(df.columns)
    ]

    # -------------------------------------------------
    # Considerar como vacío:
    # - NaN
    # - cadenas vacías
    # - espacios
    # -------------------------------------------------

    datos = df.iloc[:, columnas_filtro]

    no_vacios = datos.notna() & datos.astype(str).apply(
        lambda columna: columna.str.strip() != ""
    )

    # Una fila se conserva si tiene AL MENOS UN dato
    # en las columnas que sí nos interesan.
    filas_validas = no_vacios.any(axis=1)

    df_filtrado = df[filas_validas].copy()

    eliminadas = len(df) - len(df_filtrado)

    print(f"\nFilas conservadas : {len(df_filtrado)}")
    print(f"Filas eliminadas  : {eliminadas}")

    # -------------------------------------------------
    # Guardar
    # -------------------------------------------------

    carpeta = os.path.dirname(ruta)
    nombre = os.path.splitext(os.path.basename(ruta))[0]

    if extension == ".csv":
        salida = os.path.join(
            carpeta,
            nombre + "_filtrado.csv"
        )

        df_filtrado.to_csv(
            salida,
            index=False,
            encoding="utf-8-sig"
        )

    else:
        salida = os.path.join(
            carpeta,
            nombre + "_filtrado.xlsx"
        )

        df_filtrado.to_excel(
            salida,
            index=False
        )

    print("\n========================================")
    print("           PROCESO TERMINADO")
    print("========================================")
    print(f"Archivo generado:\n{salida}")
    print("========================================")


def main():
    ruta = obtener_archivo()

    if ruta:
        filtrar_archivo(ruta)

    input("\nPresiona ENTER para salir...")


if __name__ == "__main__":
    main()