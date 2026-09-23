#!/usr/bin/env python3
"""
Visor sencillo de archivos .tif / .tiff

Abre un cuadro de diálogo para elegir uno o varios archivos, los carga
como arrays de numpy y los muestra con matplotlib.

Requisitos:
    pip install pillow numpy matplotlib
    pip install tifffile      # opcional, pero recomendado para TIFFs científicos
"""

import sys
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import matplotlib.pyplot as plt

try:
    import tifffile
    HAY_TIFFFILE = True
except ImportError:
    HAY_TIFFFILE = False

from PIL import Image


def elegir_archivos():
    """Abre el diálogo del sistema y devuelve la lista de rutas elegidas."""
    root = tk.Tk()
    root.withdraw()          # no queremos la ventana principal, solo el diálogo
    root.attributes("-topmost", True)

    rutas = filedialog.askopenfilenames(
        title="Elige uno o varios archivos TIFF",
        filetypes=[("Imágenes TIFF", "*.tif *.tiff"), ("Todos los archivos", "*.*")],
    )
    root.destroy()
    return list(rutas)


def cargar_tif(ruta):
    """
    Devuelve (array, info) donde array es un numpy array.
    Usa tifffile si está disponible; si no, cae en Pillow.
    """
    if HAY_TIFFFILE:
        arr = tifffile.imread(ruta)
        motor = "tifffile"
    else:
        img = Image.open(ruta)
        if getattr(img, "n_frames", 1) > 1:
            frames = []
            for i in range(img.n_frames):
                img.seek(i)
                frames.append(np.array(img))
            arr = np.stack(frames)
        else:
            arr = np.array(img)
        motor = "Pillow"

    info = f"{motor} | shape={arr.shape} | dtype={arr.dtype} | min={arr.min()} max={arr.max()}"
    return arr, info


def normalizar(plano):
    """Escala a 0-1 para que se vean bien los TIFF de 16/32 bits."""
    plano = plano.astype(np.float32)
    lo, hi = np.percentile(plano, (1, 99))
    if hi <= lo:
        lo, hi = plano.min(), plano.max()
    if hi <= lo:
        return np.zeros_like(plano)
    return np.clip((plano - lo) / (hi - lo), 0, 1)


def primer_plano(arr):
    """
    Reduce un array de cualquier dimensión a algo mostrable:
    2D (gris) o 3D con 3-4 canales (color).
    """
    while arr.ndim > 3:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[-1] not in (3, 4):
        arr = arr[0]          # era un stack de planos en escala de grises
    return arr


def mostrar(ruta, arr, info):
    plano = primer_plano(arr)
    fig, ax = plt.subplots(figsize=(8, 7))

    if plano.ndim == 2:
        ax.imshow(normalizar(plano), cmap="gray")
    else:
        ax.imshow(normalizar(plano))

    ax.set_title(ruta.split("/")[-1], fontsize=11)
    ax.set_xlabel(info, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()


def main():
    # Permite pasar rutas por línea de comandos: python visor_tif.py foto.tif
    rutas = sys.argv[1:] or elegir_archivos()

    if not rutas:
        print("No elegiste ningún archivo.")
        return

    if not HAY_TIFFFILE:
        print("Aviso: tifffile no está instalado, usando Pillow. "
              "Para TIFFs científicos instala: pip install tifffile\n")

    for ruta in rutas:
        try:
            arr, info = cargar_tif(ruta)
            print(f"{ruta}\n    {info}\n")
            mostrar(ruta, arr, info)
        except Exception as e:
            print(f"Error con {ruta}: {e}")
            messagebox.showerror("Error al abrir", f"{ruta}\n\n{e}")

    plt.show()


if __name__ == "__main__":
    main()