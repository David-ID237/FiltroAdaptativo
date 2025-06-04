# Autor: Gutierrez Chavero David
# Descripción:
# Este script permite aplicar un filtro híbrido optimizado con lógica difusa a datos de series temporales,
# particularmente útiles en estudios de magnetometría geofísica.
# Incluye filtros individuales (Kalman, Mediana, ARIMA), los combina y grafica los resultados.

import argparse  # Módulo estándar para análisis de argumentos desde la línea de comandos.
import pandas as pd  # Librería de análisis de datos.
import os  # Para manipulación de rutas de archivos.

from lector_csv import cargar_csv
from filtro_kalman import FiltroKalmanOptimizado
from filtro_mediana import FiltroMedianaOptimizado
from filtro_arima import FiltroARIMAOptimizado
from hibrido import FiltroHibrido
from graficas import graficar_todo
from graficas import graficar_filtro
from graficas import terminal

# --- Análisis de argumentos por línea de comandos ---
parser = argparse.ArgumentParser(
    description="Aplica un filtro con lógica difusa a datos de magnetometría u otras series temporales.\n"
                "Permite seleccionar la columna, el rango de filas y guarda la señal filtrada."
)
parser.add_argument("--entrada", "--ent", type=str, required=True, help="Especifica el archivo CSV con los datos a cargar")
parser.add_argument("--columna", "--col", type=int, required=True, help="Columna a usar")
parser.add_argument("--fila", "-f", "--fil", type=int, default=0, help="Fila de inicio")
parser.add_argument("--filafinal", "--final", type=int, help="Fila final a usar (opcional)")
parser.add_argument("--salida", "--sal", type=str, default="senal_filtrada.csv", help="Nombre del archivo de salida ([Nombre].cvs)")
args = parser.parse_args()

# --- Carga de datos desde CSV ---
# Permite seleccionar rango de filas y una sola columna.
datos = cargar_csv(args.entrada, args.fila, args.columna, args.filafinal)

# --- Aplicación de filtros individuales ---
kalman = FiltroKalmanOptimizado(datos)
senal_kalman = kalman.ejecutar()

mediana = FiltroMedianaOptimizado(datos)
senal_mediana = mediana.ejecutar()

arima = FiltroARIMAOptimizado(datos)
senal_arima = arima.ejecutar()

# --- Graficado individual de filtros ---
# Se generan imágenes comparativas entre la señal original y la salida de cada filtro.
nombre_base = os.path.splitext(os.path.basename(args.entrada))[0]

graficar_filtro(datos, senal_kalman, args.entrada, "Kalman")
graficar_filtro(datos, senal_mediana, args.entrada, "Mediana")
graficar_filtro(datos, senal_arima, args.entrada, "ARIMA")

# --- Aplicación de filtro híbrido con lógica difusa ---
# Esta clase fue implementada por el autor y combina los filtros anteriores con ponderación lógica.
hibrido = FiltroHibrido([senal_kalman, senal_mediana, senal_arima])
senal_final = hibrido.ejecutar()

# --- Guardado del resultado final en archivo CSV ---
pd.DataFrame({"filtrada": senal_final}).to_csv(args.salida, index=False)

# --- Graficado final de la señal filtrada híbrida ---
graficar_todo(datos, senal_final, args.entrada)

# --- Finalización y reporte en terminal ---
print("\n==============================================================\n")
print("  Filtrado finalizado con éxito.\n")
terminal(senal_final)
print("==============================================================\n")
print(f"Resultados guardados en {args.salida}.")
