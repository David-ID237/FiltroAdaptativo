import numpy as np

from scipy.signal import medfilt

from genetico import AlgoritmoGenetico
from montecarlo import EvaluadorMonteCarlo

# ------------------------------------------------------------------------------
# Clase: FiltroMediana
# Descripción:
#   Aplica un filtro de mediana a una señal. Es útil para eliminar picos de ruido
#   sin afectar tanto la forma de la señal como lo haría un filtro promedio.
# Entradas:
#   - datos: señal original a filtrar.
#   - ventana: tamaño de la ventana de filtrado. Se ajusta automáticamente a impar.
# Salidas:
#   - Método aplicar() devuelve la señal suavizada.
# Notas:
#   - Se utiliza la función medfilt de SciPy, ampliamente empleada en procesamiento
#     de señales.
# ------------------------------------------------------------------------------
class FiltroMediana:
    def __init__(self, datos, ventana):
        self.datos = datos
        self.ventana = int(ventana) | 1  # Asegura impar

    def aplicar(self):
        return medfilt(self.datos, self.ventana)

# ------------------------------------------------------------------------------
# Clase: FiltroMedianaOptimizado
# Descripción:
#   Ejecuta el filtro de mediana con optimización del tamaño de la ventana mediante
#   un algoritmo genético y evaluación por simulación Monte Carlo.
# Entradas:
#   - datos: señal original a filtrar.
# Salidas:
#   - Método ejecutar() devuelve la señal suavizada con ventana óptima.
# Notas:
#   - Se busca minimizar la diferencia entre la señal filtrada y el comportamiento
#     estadístico esperado a través de evaluación Monte Carlo.
# ------------------------------------------------------------------------------
class FiltroMedianaOptimizado:
    def __init__(self, datos):
        self.datos = datos

    def ejecutar(self):
        def fitness(params):
            ventana = int(params[0]) | 1
            salida = FiltroMediana(self.datos, ventana).aplicar()
            return EvaluadorMonteCarlo.evaluar(self.datos, salida)

        genetico = AlgoritmoGenetico(bounds=[(3, 11)])  # ventanas impares
        ventana_opt = genetico.optimizar(fitness)[0]
        return FiltroMediana(self.datos, ventana_opt).aplicar()
