import numpy as np

from genetico import AlgoritmoGenetico
from montecarlo import EvaluadorMonteCarlo

class FiltroKalman:
    def __init__(self, datos, Q, R):
        self.datos = datos
        self.Q = Q * np.eye(2)  # 2x2 covariance
        self.R = R

    def aplicar(self):
        n = len(self.datos)
        x = np.array([self.datos[0], 0.0])  # estado inicial [posición, velocidad]
        P = np.eye(2)

        A = np.array([[1, 1],  # modelo de evolución constante
                      [0, 1]])
        H = np.array([[1, 0]])  # solo medimos la posición

        resultados = []

        for z in self.datos:
            # Predicción
            x_pred = A @ x
            P_pred = A @ P @ A.T + self.Q

            # Corrección
            S = H @ P_pred @ H.T + self.R
            K = P_pred @ H.T / S  # Ganancia de Kalman
            y = z - H @ x_pred    # Innovación

            x = x_pred + K.flatten() * y
            P = (np.eye(2) - K @ H) @ P_pred

            resultados.append(x[0])  # Solo guardamos la posición

        return np.array(resultados)

class FiltroKalmanOptimizado:
    def __init__(self, datos):
        self.datos = datos

    def ejecutar(self):
        def fitness(params):
            Q, R = params
            filtro = FiltroKalman(self.datos, Q, R)
            salida = filtro.aplicar()
            return EvaluadorMonteCarlo.evaluar(self.datos, salida)

        genetico = AlgoritmoGenetico(bounds=[(1e-5, 1), (1e-5, 1)])
        Q_opt, R_opt = genetico.optimizar(fitness)
        return FiltroKalman(self.datos, Q_opt, R_opt).aplicar()
