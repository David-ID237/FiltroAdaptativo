
# Filtro Adaptativo Híbrido para Señales de Magnetometría

Este proyecto implementa un sistema de filtrado adaptativo basado en lógica difusa, diseñado para señales geofísicas ruidosas obtenidas por magnetometría. Combina los filtros de Kalman, ARIMA y mediana, ajustados mediante un algoritmo genético y evaluados con métricas Monte Carlo para ofrecer un resultado óptimo. La señal final es una combinación híbrida de las salidas individuales.

## Objetivo

Reducir el ruido en señales de magnetometría utilizando una combinación inteligente de filtros, adaptando sus parámetros automáticamente para maximizar la conservación espectral, suavidad y calidad general de la señal filtrada.

## Características

- Filtros optimizados:
  - **Kalman** con ajuste de matrices Q y R
  - **ARIMA** con optimización de órdenes (p,d,q)
  - **Mediana** con selección óptima de ventana
- Algoritmo genético para búsqueda de hiperparámetros
- Evaluación robusta mediante análisis espectral, suavidad, entropía y varianza
- Lógica difusa para combinación de filtros
- Visualización automatizada de resultados

## Estructura del proyecto

```
filtro_adaptativo/
│
├── main.py              # Punto de entrada con argparse
├── lector_csv.py        # Carga y preprocesa datos CSV
├── filtro_kalman.py     # Filtro de Kalman optimizado
├── filtro_mediana.py    # Filtro de mediana optimizado
├── filtro_arima.py      # Filtro ARIMA optimizado
├── genetico.py          # Algoritmo genético para optimización
├── montecarlo.py        # Evaluación de calidad de señal
├── hibrido.py           # Combinación ponderada de filtros
└── graficas.py          # Generación de gráficas y visualización
```

## Ejecución

```bash
python main.py --entrada datos.csv --columna 1 --fila 0 --filafinal 1000 --salida salida.csv
```

Argumentos:

- `--entrada` o `--ent`: Ruta al archivo CSV de entrada
- `--columna` o `--col`: Número de columna con la señal a filtrar
- `--fila` o `-f`: Fila de inicio (por defecto 0)
- `--filafinal` o `--final`: Fila final (opcional)
- `--salida` o `--sal`: Nombre del archivo CSV con la señal filtrada

## Requisitos

- Python 3.8+
- Pandas
- NumPy
- SciPy
- Statsmodels
- Matplotlib

Instalar dependencias con:

```bash
pip install -r requirements.txt
```

## Aplicación

Este filtro fue desarrollado como parte de un proyecto académico de procesamiento de señales para aplicaciones geofísicas. Permite conservar contenido espectral útil, eliminar ruido y facilitar la interpretación geocientífica.

## Resultados esperados

Al ejecutar el programa se generan:
- Un archivo CSV con la señal filtrada
- Gráficas:
  - Señal original
  - Señales filtradas por Kalman, ARIMA, Mediana
  - Señal híbrida combinada
