import numpy as np

from scipy.signal import medfilt

from genetico import AlgoritmoGenetico
from montecarlo import EvaluadorMonteCarlo

class FiltroMediana:
    def __init__(self, datos, ventana):
        self.datos = datos
        self.ventana = int(ventana) | 1  # Asegura impar

    def aplicar(self):
        return medfilt(self.datos, self.ventana)

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
