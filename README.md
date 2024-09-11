# Análisis de Retrasos de Vuelos

Este repositorio contiene un conjunto de datos de vuelos y un script de análisis en Jupyter Notebook para estudiar los factores que afectan los retrasos de vuelos. El objetivo principal es predecir si un vuelo se retrasará más de 3 horas y analizar los tipos de retrasos más comunes en esas circunstancias.

## Contenidos

1. **`data.csv`**: Archivo CSV con datos de vuelos.
2. **`flight_delay_analysis.ipynb`**: Script Jupyter Notebook que realiza el análisis de datos, preprocesamiento y modelado para la predicción de retrasos de vuelos.

## Dataset

El archivo `data.csv` contiene la siguiente información sobre los vuelos:

- **DayOfWeek**: Día de la semana (1 = Lunes, 7 = Domingo)
- **Date**: Fecha programada
- **DepTime**: Hora de salida real (local, hhmm)
- **ArrTime**: Hora de llegada real (local, hhmm)
- **CRSArrTime**: Hora de llegada programada (local, hhmm)
- **UniqueCarrier**: Código del transportista
- **Airline**: Compañía aérea
- **FlightNum**: Número de vuelo
- **TailNum**: Número de cola del avión
- **ActualElapsedTime**: Tiempo real en el aire (en minutos) con TaxiIn/Out
- **CRSElapsedTime**: Tiempo estimado de vuelo (en minutos)
- **AirTime**: Tiempo de vuelo (en minutos)
- **ArrDelay**: Diferencia en minutos entre la hora de llegada programada y real
- **Origin**: Código IATA del aeropuerto de origen
- **Org_Airport**: Nombre del aeropuerto de origen
- **Dest**: Código IATA del aeropuerto de destino
- **Dest_Airport**: Nombre del aeropuerto de destino
- **Distance**: Distancia entre aeropuertos (millas)
- **TaxiIn**: Tiempo de llegada y llegada a la puerta del aeropuerto de destino, en minutos
- **TaxiOut**: Tiempo transcurrido entre la salida del aeropuerto de origen y el despegue, en minutos
- **Cancelled**: ¿Se canceló el vuelo? 1 = sí, 0 = no
- **CancellationCode**: Razón de la cancelación
- **Diverted**: ¿Se redireccionó el vuelo? 1 = sí, 0 = no
- **CarrierDelay**: Retraso por parte del transportista (en minutos)
- **WeatherDelay**: Retraso debido al clima (en minutos)
- **NASDelay**: Retraso por parte del sistema nacional de aviación (en minutos)
- **SecurityDelay**: Retraso debido a seguridad (en minutos)
- **LateAircraftDelay**: Retraso por este motivo (en minutos)

## Requisitos

Para ejecutar el Jupyter Notebook, necesitarás tener instalados los siguientes paquetes en tu entorno Python:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

Puedes instalar estos paquetes usando pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Cómo Ejecutar

1. Clona este repositorio:
```bash
git clone https://github.com/nevalego/flights-delay.git
```
2. Navega al directorio del repositorio:
```bash
cd flights-delay
```
3. Abre el Jupyter Notebook:
```bash
jupyter notebook flight_delay_analysis.ipynb
```

## Descripción del Análisis

### 1. Caracterización  del Dataset

    1. Muestra las primeras filas del dataset: Se inspecciona una muestra inicial de los datos para tener una visión general.
    2. Información general y descripción estadística: Se proporciona información detallada sobre los tipos de datos y una descripción estadística de las variables numéricas.
    3. Número de clases de la variable objetivo: Se determina el número de clases para la variable objetivo (retraso > 3 horas) y el tipo de valores que toma.
    4. Número total de instancias: Se muestra el tamaño total del dataset.
    5. Valores ausentes: Se identifican y cuantifican los valores ausentes en el dataset.


### 2. Análisis Exploratorio de Datos (EDA)

    1. Distribución de los retrasos de llegada: Se visualiza cómo se distribuyen los retrasos de llegada mediante histogramas y gráficos de densidad.
    2. Retrasos de llegada por clases de retraso: Se analizan los retrasos de llegada en función de si el retraso supera las 3 horas o no.
    3. Relación entre retrasos y distancia: Se explora la relación entre la distancia del vuelo y el retraso de llegada utilizando gráficos de dispersión.
    4. Tiempo real vs tiempo estimado de vuelo: Se compara el tiempo real de vuelo con el tiempo estimado para evaluar la precisión de las estimaciones.



### 3. Preprocesamiento de Datos

    1. Tratamiento de valores ausentes: Se reemplazan los valores ausentes con la moda de cada columna.
    2. Tratamiento de valores duplicados: Se eliminan las filas duplicadas para asegurar la calidad del dataset.
    3. Codificación de variables categóricas: Se codifican las variables categóricas en valores numéricos para su uso en el modelado.


### 4. Modelado

    1. División del dataset: El dataset se divide en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba).
    2. Entrenamiento del modelo de regresión lineal: Se entrena un modelo de regresión lineal utilizando el conjunto de entrenamiento.
    3. Evaluación del modelo: Se evalúa el rendimiento del modelo utilizando métricas como el error cuadrático medio (MSE) y el coeficiente de determinación (R^2). Se comparan las predicciones del modelo con los valores reales en un gráfico.