# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 19:15:04 2024

@author: KevinRosero

Modelos de clasificación: Bosques aleatorios
cOMBINA LA POTENCIA DE DIFERENTES ALGORITMOS DE MACHINE LEARNING
clasificacion de votos por mayoria

Proces:
    
    1.- Seleccionar un numero aleatorio de K puntos del Conjunto de Entrenamiento
    2.- De esos punto se construye un arbol de decision a esos K puntos de datos
    3.- Elegimos un número NTree de árboles que queremos construir y repetir los pasos
    4.-Para clasificar un nuevo punto, hacer que cada uno de los NTree arboles elaborados a
    que categoria pertenece y asignar el nuevo punto a la categoria con mas votos
    
    Por default este algoritmo en python toma 10 arboles
    
    
"""

##En algun lado va a aparecer la palabra 'clasificación'

## Importar las librearías
import numpy as np ## Libraría para tratamiento de números y herramientas matemáticas (np Alias)
import matplotlib.pyplot as plt  ## Librería para representación gráfica (pyplot es una sublibrearía - gráficos) (plt Alias)
import pandas as pd  ## Librería para manipular datos (pd Alias)
##Ahora si es mas importante el tema de matplotlib por que se vuelve mas complejo


dataset= pd.read_csv("Social_Network_Ads.csv")

#Establecemos las variables independientes Se pone en arboles de decision y Random Forest
X = dataset.iloc[:, [2,3]].values


#Establecemos la variable dependiente no te olvides de poner el values
y = dataset.iloc[:, 4].values


##mDividimos el Dataset en entrenamiento y test

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

##Escalado
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)



## Se importa el tree que es el clasificador de arboles de decision averiguar que es el indice Jimmy
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=0)
rf.fit(X_train, y_train)


## Predicción de los resultados con el conjunto de test
y_pred = rf.predict(X_test)


from sklearn.metrics import confusion_matrix
conmat = confusion_matrix(y_test, y_pred)


### Visualizar el algotirmo de train graficamente con los resultados
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2 , rf.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                alpha = 0.75, cmap = ListedColormap(('red', 'green')))                     
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], 
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title("Random Forest Train")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()

### Visualizar el algotirmo de test graficamente con los resultados
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2 , rf.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                alpha = 0.75, cmap = ListedColormap(('red', 'green')))                     
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], 
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title("Random Forest Test")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()