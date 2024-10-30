# -*- coding: utf-8 -*-
"""
Created on Wed Oct-16  18:14:30 2024

@author: Kevin Rosero

MULTIPLE LINEAR REGRESSION
Restricciones:
    - Linealidad
    - Homocedasticidad
    - Normalidad multivariable
    - Independencia de errores
    - Ausencia de multicolinealidad (si hay más de una variable dummy, eliminar una)
"""

'''
Métodos para construir modelos multivariable:
    - Exhaustivo: Usar todas las variables en el modelo.
    - Eliminación hacia atrás: Quitar variables con p-valor alto hasta que todas cumplan.
    - Selección hacia adelante: Agregar variables con p-valor bajo hasta el límite.
    - Eliminación bidireccional: Combina eliminación hacia atrás y adelante.
    - Comparación de Scores: Selección del mejor modelo usando el criterio de Akaike.
'''

## Análisis de startups para decidir inversión según criterios y analizar relación entre ganancias y ubicación
## Predicción de ingresos de la empresa (Y)

## Importación de librerías
import numpy as np  # Para operaciones numéricas
import matplotlib.pyplot as plt  # Para gráficos
import pandas as pd  # Para manipulación de datos

## Cargar datos
dataset = pd.read_csv("50_Startups.csv")

## Definir variables independientes (X) y dependientes (y)
X = dataset.iloc[:, :-1].values  # Todas las filas y columnas menos la última
y = dataset.iloc[:, 4].values  # Columna de ingresos

'''

## Tratamiento de valores faltantes (NAN)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")  # Sustitución de NAN por la media
imputer = imputer.fit(X[:, 1:3])  # Aplicación en columnas seleccionadas
X[:, 1:3] = imputer.transform(X[:, 1:3])  # Reemplazo de valores faltantes

## Codificación de datos categóricos
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])  # Conversión de datos categóricos a numéricos

'''

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])  # Conversión de datos categóricos

## Aplicación de OneHotEncoder para variables dummy
onehotencoder = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [3])], remainder='passthrough')
X = np.array(onehotencoder.fit_transform(X), dtype=float)  # Matriz de características

'''
## Eliminar variables dummy adicionales
X = X[:, 1:]  # Eliminación de la primera columna de variables dummy

'''

## División del conjunto de datos en entrenamiento y prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



## Modelo de regresión múltiple
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)  # Entrenamiento del modelo

## Predicción y prueba del modelo
y_pred = regression.predict(X_test)

## Eliminación hacia atrás para optimizar el modelo
import statsmodels.api as sm
X = np.append(arr=np.ones((50,1)).astype(int), values=X, axis=1)  # Añade columna de unos para el intercepto
SL = 0.05  # Nivel de significancia

# Ajuste del modelo eliminando variables con p-valor alto
X_opt = X[:, [0,1,2,3,4,5]]
regression_OLS = sm.OLS(y, X_opt).fit()
regression_OLS.summary()

X_opt = X[:, [0,1,3,4,5]]
regression_OLS = sm.OLS(y, X_opt).fit()
regression_OLS.summary()

X_opt = X[:, [0,3,4,5]]
regression_OLS = sm.OLS(y, X_opt).fit()
regression_OLS.summary()

X_opt = X[:, [0,3,5]]
regression_OLS = sm.OLS(y, X_opt).fit()
regression_OLS.summary()

X_opt = X[:, [0,3]]
regression_OLS = sm.OLS(y, X_opt).fit()
regression_OLS.summary()

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
