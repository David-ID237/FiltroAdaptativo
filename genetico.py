import numpy as np

from scipy.optimize import differential_evolution

# ------------------------------------------------------------------------------
# Clase: AlgoritmoGenetico
# Descripción:
#   Clase que encapsula un proceso de optimización global basado en la evolución
#   diferencial, utilizado para ajustar hiperparámetros de filtros de señal.
#
# Métodos:
#   - __init__(bounds):
#       Recibe los límites de búsqueda [(min1, max1), ..., (minN, maxN)] para
#       cada uno de los parámetros a optimizar.
#
#   - optimizar(fitness):
#       Ejecuta la optimización global utilizando el método
#       scipy.optimize.differential_evolution, con un número fijo de iteraciones.
#       Retorna los parámetros óptimos encontrados que minimizan la función fitness.
#
# Notas:
#   - Aunque el nombre de la clase es "AlgoritmoGenetico", en realidad emplea
#     evolución diferencial, una técnica de optimización estocástica de la misma
#     familia evolutiva.
# ------------------------------------------------------------------------------
class AlgoritmoGenetico:
    def __init__(self, bounds):
        self.bounds = bounds

    def optimizar(self, fitness):
        resultado = differential_evolution(fitness, self.bounds, maxiter=50, disp=False)
        return resultado.x
