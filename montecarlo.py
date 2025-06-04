import numpy as np

from scipy.fft import fft
from scipy.stats import entropy

# ------------------------------------------------------------------------------
# Clase: EvaluadorMonteCarlo
# Descripción:
#   Evalúa la calidad de una señal filtrada mediante un conjunto de métricas
#   cuantitativas, combinadas en una única puntuación. Esta evaluación guía
#   el proceso de optimización de los filtros usando simulaciones tipo
#   Monte Carlo.
#
# Métodos:
#   - evaluar(original, filtrada):
#       Calcula un puntaje compuesto considerando similitud espectral, suavidad,
#       entropía y varianza. Penaliza señales demasiado planas o con pérdida de
#       información espectral relevante.
#
#   - promedio_metricas(lista_resultados):
#       Calcula el promedio de una lista de evaluaciones.
#
# Notas:
#   - Las métricas utilizadas combinan heurísticas comunes en análisis de
#     señales (FFT, entropía de Shannon, derivadas, varianza).
# ------------------------------------------------------------------------------
class EvaluadorMonteCarlo:
    @staticmethod
    def evaluar(original, filtrada):
        # Normalizar espectros
        espectro_original = np.abs(fft(original))
        espectro_filtrada = np.abs(fft(filtrada))

        espectro_original /= np.sum(espectro_original) + 1e-12
        espectro_filtrada /= np.sum(espectro_filtrada) + 1e-12

        # Diferencia espectral con pesos (más peso a bajas frecuencias)
        pesos = np.linspace(1, 0.1, len(espectro_original))
        delta_espectro = espectro_original - espectro_filtrada
        espectro_score = np.sum(pesos * delta_espectro**2)

        # Suavidad: suma de segundas derivadas absolutas
        suavidad = np.sum(np.abs(np.diff(filtrada, 2)))

        # Entropía del histograma
        hist, _ = np.histogram(filtrada, bins=50, density=True)
        ent = entropy(hist + 1e-8)

        # Penalización por ser plano (varianza casi nula)
        varianza = np.var(filtrada)
        penalizacion_plano = 1.0 / (varianza + 1e-8)

        # Evaluación total ponderada
        return (
            1.0 * espectro_score +    # Conservación de contenido espectral
            0.3 * suavidad +          # Suavidad (evitar señales ruidosas)
            0.5 * ent +               # Entropía (evita señal excesivamente uniforme)
            0.2 * penalizacion_plano  # Penaliza señales demasiado planas
        )

    @staticmethod
    def promedio_metricas(lista_resultados):
        return np.mean(lista_resultados)
