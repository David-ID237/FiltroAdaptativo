import pandas as pd
import numpy as np

def cargar_csv(archivo, fila, columna, final):
    df = pd.read_csv(archivo, skiprows=fila)
    datos = df.iloc[:final, columna].values if final else df.iloc[:, columna].values
    return datos.astype(np.float32)
