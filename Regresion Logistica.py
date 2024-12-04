# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 19:08:04 2024

@author: KevinRosero
"""

##En algun lado va a aparecer la palabra 'clasificación'

## Importar las librearías
import numpy as np ## Libraría para tratamiento de números y herramientas matemáticas (np Alias)
import matplotlib.pyplot as plt  ## Librería para representación gráfica (pyplot es una sublibrearía - gráficos) (plt Alias)
import pandas as pd  ## Librería para manipular datos (pd Alias)
##Ahora si es mas importante el tema de matplotlib por que se vuelve mas complejo


dataset  = pd.read_csv("Social_Network_Ads.csv")


#Establecemos las variables independientes
X = dataset.iloc[:, [2,3]]


#Establecemos la variable dependiente
y = dataset.iloc[:, -1]


##mDividimos el Dataset en entrenamiento y test

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

##Escalado
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

## Ajustar el modelo de Regresión Logistica en el Conjunto de entrenamiento
##Aprende a predecir las clasificaciones
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)


#Predicción de los resultados
y_pred = classifier.predict(X_test)


## Analisis de resultados con la matriz de confusion
## Saber si la prediccion es buena
## Se calcula la matriz de confusion sobre el modelo de testing
## Cuantas son las predicciones correctas

from sklearn.metrics import confusion_matrix
conmet = confusion_matrix(y_test, y_pred)

## 65 no compraron 24 si compraron
## 8 y 3 son correctas
## Cual es el porcentaje??

## Visualizar el algoritmo de train graficamente con los esultados
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(
    np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
    np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01)
)

# Visualizar las regiones de clasificación
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))

# Limitar los ejes
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

# Graficar los puntos de datos (observaciones)
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)

# Añadir etiquetas y título
plt.title('Clasificación (Conjunto de Entrenamiento)')
plt.xlabel('Edad')
plt.ylabel('Sueldo estimado')
plt.legend()
plt.show()


## Visualizar el algoritmo de train graficamente con los esultados
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(
    np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
    np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01)
)

# Visualizar las regiones de clasificación
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))

# Limitar los ejes
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

# Graficar los puntos de datos (observaciones)
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)

# Añadir etiquetas y título
plt.title('Clasificación (Conjunto de Entrenamiento)')
plt.xlabel('Edad')
plt.ylabel('Sueldo estimado')
plt.legend()
plt.show()



