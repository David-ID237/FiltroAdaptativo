import numpy as np
import warnings

from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA

from genetico import AlgoritmoGenetico
from montecarlo import EvaluadorMonteCarlo

# Ignorar advertencias de convergencia de ARIMA
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
warnings.simplefilter("ignore", ConvergenceWarning)

class FiltroARIMA:
    def __init__(self, datos, p, d, q):
        self.datos = datos
        self.p = int(p)
        self.d = int(d)
        self.q = int(q)

    def aplicar(self):
        modelo = ARIMA(self.datos, order=(self.p, self.d, self.q))
        resultado = modelo.fit()
        return resultado.fittedvalues

class FiltroARIMAOptimizado:
    def __init__(self, datos):
        self.datos = datos

    def ejecutar(self):
        def fitness(params):
            p, d, q = map(int, params)
            salida = FiltroARIMA(self.datos, p, d, q).aplicar()
            return EvaluadorMonteCarlo.evaluar(self.datos, salida)

        genetico = AlgoritmoGenetico(bounds=[(1, 5), (0, 2), (0, 5)])
        p_opt, d_opt, q_opt = genetico.optimizar(fitness)
        return FiltroARIMA(self.datos, p_opt, d_opt, q_opt).aplicar()
