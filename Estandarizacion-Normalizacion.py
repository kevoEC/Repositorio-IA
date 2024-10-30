# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 19:48:09 2024

@author: KevinRosero

"""

import numpy as np #numericos
import matplotlib.pyplot as plt #graficos
import pandas as pd # analisis de datos

dataset = pd.read_csv("Data.csv")

X = dataset.iloc[:, :-1].values # TOMA TODAS LAS FILAS Y PYTON HACE UN N-1 y

Y = dataset.iloc[:,3].values # TOMA TODAS LAS FILAS Y COGE LA COLUMNA 3


from sklearn.impute import SimpleImputer #Son librerias enormes y solo necesito exportar ciertas sublibrerias
# cuando se habla de imputacion es que voy a estandarizar o normalizar

imputer = SimpleImputer(missing_values = np.nan, strategy = "mean")
##imputer.fit(X) ## Con esto es la media a todas las columnas

imputer = imputer.fit(X[:, 1:3])

X[:, 1:3] = imputer.transform(X[:, 1:3])


from sklearn import preprocessing

labelencoder_X = preprocessing.LabelEncoder();
## Transforma las categoricas en numeros, en este caso nombres de paises en numeros

X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
# Si mando asi no me va a enteder, debo dicotomizar, poner entre 0 y unos
# One Hot Encoder, voy a tener tantas columnas como valores tenga

onehotencoder = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [0])], remainder='passthrough')

X = np.array(onehotencoder.fit_transform(X), dtype=float)

#label encoder pone etiquetas a variables dicotomicas osea categoricas

labelencoder_y = LabelEncoder()
y =  labelencoder_y.fit_transform(Y)


# Voy a dividir los datos entre entrenamiento y test

from sklearn.model_selection import train_test_split

#Reparte el 80% a train que es a aprender y 20% a test para validar si aprendio, 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
