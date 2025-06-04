import pandas as pd
import numpy as np

# ------------------------------------------------------------------------------
# Función: cargar_csv
# Descripción:
#   Carga datos numéricos desde un archivo CSV, seleccionando una columna específica
#   y un rango opcional de filas. Utiliza pandas para facilitar la lectura estructurada.
# Entradas:
#   - archivo: ruta al archivo CSV.
#   - fila: número de fila inicial a partir de la cual comenzar la lectura.
#   - columna: índice de la columna que contiene los datos deseados.
#   - final: número de fila final a utilizar (opcional; si es None o 0, se usa hasta el final).
# Salidas:
#   - Un arreglo numpy (float32) con los datos extraídos.
# ------------------------------------------------------------------------------
def cargar_csv(archivo, fila, columna, final):
    df = pd.read_csv(archivo, skiprows=fila)
    datos = df.iloc[:final, columna].values if final else df.iloc[:, columna].values
    return datos.astype(np.float32)
