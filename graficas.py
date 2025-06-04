import os

from matplotlib import pyplot as plt

def graficar_todo(original, filtrada, nombre_archivo):
    prefijo = os.path.splitext(os.path.basename(nombre_archivo))[0]

    # Señal original
    plt.figure(figsize=(10, 4))
    plt.plot(original, label="Original", color='blue')
    plt.title("Señal Original")
    plt.grid(True)
    plt.savefig(f"{prefijo}_original.png")
    plt.close()

    # Señal filtrada
    plt.figure(figsize=(10, 4))
    plt.plot(filtrada, label="Filtrada", color='green')
    plt.title("Señal Filtrada")
    plt.grid(True)
    plt.savefig(f"{prefijo}_filtrada.png")
    plt.close()

    # Superposición
    plt.figure(figsize=(10, 4))
    plt.plot(original, label="Original", alpha=0.6)
    plt.plot(filtrada, label="Filtrada", alpha=0.7)
    plt.title("Superposición de Señales")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{prefijo}_comparacion.png")
    plt.close()

def graficar_filtro(original, filtrada, nombre_archivo, etiqueta):
    prefijo = os.path.splitext(os.path.basename(nombre_archivo))[0]

    plt.figure(figsize=(10, 4))
    plt.plot(original, label="Original", alpha=0.5)
    plt.plot(filtrada, label=etiqueta, alpha=0.7)
    plt.title(f"Filtro {etiqueta}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{prefijo}_{etiqueta.lower()}.png")
    plt.close()

def terminal(senal, ancho=60, alto=10):
    import numpy as np

    senal = np.array(senal)
    senal_norm = (senal - np.min(senal)) / (np.max(senal) - np.min(senal) + 1e-8)
    senal_reducida = senal_norm[::max(1, len(senal) // ancho)]

    for nivel in reversed(range(alto)):
        linea = ''
        umbral = nivel / (alto - 1)
        for valor in senal_reducida:
            linea += '█' if valor >= umbral else ' '
        print(linea)