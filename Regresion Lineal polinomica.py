# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 19:14:22 2024

@author: KevinRosero
"""

## Importar las librearías
import numpy as np ## Libraría para tratamiento de números y herramientas matemáticas (np Alias)
import matplotlib.pyplot as plt  ## Librería para representación gráfica (pyplot es una sublibrearía - gráficos) (plt Alias)
import pandas as pd  ## Librería para manipular datos (pd Alias)

dataset  = pd.read_csv("Position_Salaries.csv")

#Establecemos el nivel como X
X = dataset.iloc[:, 1:2]


#Establecemos el salario como y
y = dataset.iloc[:, -1]


#En este caso no es necesario dividir en entrnamiento y test pues tenemos valores lineales y que no se repiten
#Ajustar el modelo con Regresion linea
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
##Ajustar los datos
lin_reg.fit(X,y)

#Ajustar el modelo con Regresion Polinomial
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)#Sin el degree automaticamente
X_poly = poly_reg.fit_transform(X)##A mi X estoy ajustando para que se ajuste al grado por lo que se crea n columnas en base a los grados


###Objeto Polinomial transformado
lin_reg2 = LinearRegression()#Todos los modelos estan basados en la Regresion lineal
lin_reg2.fit(X_poly, y)#


## Visualización en lineal (NO SE DEBE HACER)
plt.scatter(X, y, color = "red")
plt.plot(X, lin_reg.predict(X), color = "blue")
plt.title("Modelo de regresion Linear")
plt.xlabel("Posicion del empleado")
plt.ylabel("Sueldo ($)")
plt.show()

## Visualización en polinomial
plt.scatter(X, y, color = "red")
plt.plot(X, lin_reg2.predict(X_poly), color = "blue")
plt.title("Modelo de regresion polinomial")
plt.xlabel("Posicion del empleado")
plt.ylabel("Sueldo ($)")
plt.show()


###Este es el nuevo codigo - se usa para ingresar variables
X_grid = np.arange(min(X ['Level']), max(X['Level']), 0.1, dtype=float )
X_grid = X_grid.reshape((len(X_grid)), 1)


## Visualización en polinomial
plt.scatter(X, y, color = "red")
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid) ), color = "blue")
plt.title("Modelo de regresion polinomia X_grid")
plt.xlabel("Posicion del empleado")
plt.ylabel("Sueldo ($)")
plt.show()


##Prediccion de nuestros modelos
lin_reg.predict([[6.5]])


lin_reg2.predict(poly_reg.fit_transform([[6.5]]))
