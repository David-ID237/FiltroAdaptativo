# Autor: Gutierrez Chavero David
# Este módulo contiene funciones para graficar señales originales y filtradas,
# así como una visualización ASCII en terminal mostrada al finalizar el proceso.

import os   # Módulo estándar para manejo de archivos y rutas

from matplotlib import pyplot as plt    # Librería de gráficos 2D

# ------------------------------------------------------------------------------
# Función: graficar_todo
# Descripción:
#   Genera y guarda tres gráficos:
#     - la señal original,
#     - la señal filtrada final,
#     - y una superposición de ambas.
# Entradas:
#   - original: lista o array con los datos crudos.
#   - filtrada: lista o array con la señal ya procesada.
#   - nombre_archivo: ruta del archivo de entrada, se usa para nombrar las imágenes.
# ------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------
# Función: graficar_filtro
# Descripción:
#   Genera un gráfico comparativo entre la señal original y la salida de un filtro
#   individual (Kalman, Mediana, ARIMA, etc.).
# Entradas:
#   - original: datos sin filtrar.
#   - filtrada: salida del filtro aplicado.
#   - nombre_archivo: nombre del archivo de entrada original.
#   - etiqueta: nombre del filtro para el título y el archivo de imagen.
# ------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------
# Función: terminal
# Descripción:
#   Muestra una versión reducida de la señal en formato gráfico ASCII tipo "sparkline"
#   directamente en la terminal.
# Entradas:
#   - senal: lista o array de la señal.
#   - ancho: número de columnas en el gráfico ASCII (default: 60)
#   - alto: número de niveles verticales en caracteres (default: 10)
# Notas:
#   - Inspirado en representaciones tipo "sparkline" comúnmente usadas en UNIX y Python.
#     No basado en una librería específica, desarrollado desde cero.
# ------------------------------------------------------------------------------
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