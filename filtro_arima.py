import numpy as np
import warnings

from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA

from genetico import AlgoritmoGenetico
from montecarlo import EvaluadorMonteCarlo

# Ignorar advertencias de convergencia de ARIMA
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
warnings.simplefilter("ignore", ConvergenceWarning)

# ------------------------------------------------------------------------------
# Clase: FiltroARIMA
# Descripción:
#   Implementa un modelo ARIMA (AutoRegresivo Integrado de Media Móvil) para
#   suavizado de series temporales. Ajusta el modelo a los datos y devuelve
#   los valores ajustados como señal filtrada.
# Entradas:
#   - datos: señal original (serie temporal).
#   - p, d, q: parámetros del modelo ARIMA.
# Salidas:
#   - Método aplicar() devuelve la señal suavizada con el modelo ajustado.
# Notas:
#   - Se usa la clase ARIMA de la biblioteca statsmodels.
#   - El modelo ARIMA es un enfoque estadístico común y probado
#     en análisis de series temporales.
#   - Se ignoran advertencias de convergencia para evitar ruido en consola.
# ------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------
# Clase: FiltroARIMAOptimizado
# Descripción:
#   Ajusta automáticamente los parámetros (p, d, q) del modelo ARIMA usando un
#   algoritmo genético y evaluación basada en simulaciones Monte Carlo.
# Entradas:
#   - datos: señal original (serie temporal).
# Salidas:
#   - Método ejecutar() devuelve la señal suavizada con parámetros óptimos.
# Notas:
#   - Esta clase automatiza el uso de ARIMA sin intervención manual,
#     como parte del enfoque híbrido.
#   - Utiliza un optimizador evolutivo (genético) y validación estocástica.
# ------------------------------------------------------------------------------
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
