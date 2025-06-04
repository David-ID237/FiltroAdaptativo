import numpy as np

from genetico import AlgoritmoGenetico
from montecarlo import EvaluadorMonteCarlo

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
