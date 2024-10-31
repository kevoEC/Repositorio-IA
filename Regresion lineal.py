# -*- coding: utf-8 -*-
"""

Kevin Rosero
SIMPLE LINEAR REGRESSION
"""

## Análisis de salarios en función de años de experiencia
## Predicción de sueldos mediante un modelo lineal con datos de "Salary_Data.csv"

## Importación de librerías
import numpy as np  # Para operaciones numéricas
import matplotlib.pyplot as plt  # Para gráficos
import pandas as pd  # Para manipulación de datos

## Cargar datos
dataset = pd.read_csv("Salary_Data.csv")

## Definir las variables independientes (X) y dependientes (y)
X = dataset.iloc[:, :-1].values  # Selección de todas las filas y columnas excepto la última
y = dataset.iloc[:, 1].values  # Segunda columna (sueldo)

'''

## Manejo de valores faltantes (NAN)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")  # Sustituye NAN por la media
imputer = imputer.fit(X[:, 1:3])  # Se aplica en columnas seleccionadas
X[:, 1:3] = imputer.transform(X[:, 1:3])  # Asigna los valores calculados

## Codificación de variables categóricas
from sklearn import preprocessing
labelencoder_X = preprocessing.LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])  # Codificación de datos categóricos

from sklearn.preprocessing import OneHotEncoder, LabelEncoder  
from sklearn.compose import ColumnTransformer
onehotencoder = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(onehotencoder.fit_transform(X), dtype=float)

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

'''

## División de los datos en conjuntos de entrenamiento y prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0) 

## Escalado de datos
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)  # El conjunto de prueba se escala igual que el entrenamiento

## Modelo de regresión
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)  # Entrenamiento del modelo

## Predicción y prueba del modelo
y_pred = regression.predict(X_test)

## Visualización de resultados (entrenamiento)
plt.scatter(X_train, y_train, color="red")
plt.plot(X_train, regression.predict(X_train), color="blue")
plt.title("Sueldo vs Años de Experiencia (Entrenamiento)")
plt.xlabel("Años de Experiencia")
plt.ylabel("Sueldo ($)")
plt.show()

## Visualización de resultados (prueba)
plt.scatter(X_test, y_test, color="red")
plt.plot(X_test, regression.predict(X_test), color="blue")
plt.title("Sueldo vs Años de Experiencia (Prueba)")
plt.xlabel("Años de Experiencia")
plt.ylabel("Sueldo ($)")
plt.show()
