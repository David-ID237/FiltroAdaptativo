import numpy as np

from genetico import AlgoritmoGenetico
from montecarlo import EvaluadorMonteCarlo

# ------------------------------------------------------------------------------
# Clase: FiltroHibrido
# Descripción:
#   Implementa una combinación ponderada de varias señales filtradas (por ejemplo,
#   Kalman, ARIMA, mediana), ajustando automáticamente los pesos óptimos mediante
#   evolución diferencial.
#
# Métodos:
#   - __init__(señales):
#       Recibe una lista de señales filtradas (arrays NumPy) que se desean combinar.
#
#   - ejecutar():
#       Realiza la combinación de las señales mediante una mezcla ponderada.
#       Los pesos de mezcla se optimizan utilizando una estrategia evolutiva para
#       minimizar una función de evaluación espectro-temporal basada en Monte Carlo.
#       Devuelve la señal combinada optimizada.
#
# Detalles del algoritmo:
#   - Los pesos son restringidos a ser positivos y normalizados para sumar 1.
#   - La señal combinada se calcula como suma ponderada: ∑(wᵢ * señalᵢ).
#   - La evaluación se hace con respecto a la primera señal (señales[0]) como base.
#
# Notas:
#   - Aunque el método usa el nombre "genético", internamente emplea evolución
#     diferencial por su eficacia en espacios de búsqueda continuos.
# ------------------------------------------------------------------------------
class FiltroHibrido:
    def __init__(self, señales):
        self.señales = señales

    def ejecutar(self):
        def fitness(pesos):
            pesos = np.abs(pesos)
            pesos /= np.sum(pesos)
            mezcla = sum(w * s for w, s in zip(pesos, self.señales))
            return EvaluadorMonteCarlo.evaluar(self.señales[0], mezcla)

        genetico = AlgoritmoGenetico(bounds=[(0, 1)] * len(self.señales))
        pesos_opt = genetico.optimizar(fitness)
        pesos_opt = np.abs(pesos_opt)
        pesos_opt /= np.sum(pesos_opt)
        return sum(w * s for w, s in zip(pesos_opt, self.señales))
