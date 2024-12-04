# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 19:20:03 2024

@author: KevinRosero
"""

##En algun lado va a aparecer la palabra 'clasificación'

## Importar las librearías
import numpy as np ## Libraría para tratamiento de números y herramientas matemáticas (np Alias)
import matplotlib.pyplot as plt  ## Librería para representación gráfica (pyplot es una sublibrearía - gráficos) (plt Alias)
import pandas as pd  ## Librería para manipular datos (pd Alias)
##Ahora si es mas importante el tema de matplotlib por que se vuelve mas complejo


dataset= pd.read_csv("Social_Network_Ads.csv")


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

## Ajuste del clasificador en el conjunto de entrenamiento
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2 )
knn.fit(X_train, y_train)
 

## Predicción de los resultados con el conjunto de test
y_pred = knn.predict(X_test)


## Analisis de resultados con la matriz de confusion
## Saber si la prediccion es buena
## Se calcula la matriz de confusion sobre el modelo de testing
## Cuantas son las predicciones correctas

from sklearn.metrics import confusion_matrix
conmet = confusion_matrix(y_test, y_pred)

## 65 no compraron 24 si compraron
## 8 y 3 son correctas
## Cual es el porcentaje??

### Visualizar el algotirmo de train graficamente con los resultados
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2 , knn.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                alpha = 0.75, cmap = ListedColormap(('red', 'green')))                     
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], 
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title("K-NN (Training set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()
 
 
### Visualizar el algotirmo de test graficamente con los resultados
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2 , knn.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                alpha = 0.75, cmap = ListedColormap(('red', 'green')))                     
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], 
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title("K-NN (Test set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()