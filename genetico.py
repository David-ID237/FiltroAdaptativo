import numpy as np

from scipy.optimize import differential_evolution

class AlgoritmoGenetico:
    def __init__(self, bounds):
        self.bounds = bounds

    def optimizar(self, fitness):
        resultado = differential_evolution(fitness, self.bounds, maxiter=50, disp=False)
        return resultado.x
