# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 10:28:44 2020

@author: vicente candela perez
"""

import pandas as pd

pokemon = pd.read_csv("C:/Users/vicente candela pere/Desktop/3º de carrera/Mineria de Datos/practicas/Practica 3/Pokemon.csv", sep = ",")

descriptive = pokemon.describe()

pokemon["Generation"].unique()
pokemon["Legendary"].unique()

pokemon["Generation"].value_counts()
pokemon["Legendary"].value_counts()


%matplotlib auto
import seaborn as sns

sns.pairplot(pokemon.select_dtypes(exclude=[object]))

# preprocesamiento de datos
# seleccionar solo las variables numericas otra forma

num_pokemon = pokemon.drop(["Name", "Type 1", "Type 2", "Generation"], axis = 1)

num_pokemon.loc[num_pokemon.loc[:, "Legendary"] == "VERDADERO", "Legendary"] = 1 
num_pokemon.loc[num_pokemon.loc[:, "Legendary"] == "FALSO", "Legendary"] = 0 

num_pokemon["Legendary"].unique()


#clasificacion
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt


X = num_pokemon.values[:, :-1] #Variables explicativas
y = num_pokemon.values[:, -1] #variable objetivo
y = y.astype("int")

modelo = DecisionTreeClassifier(criterion='entropy', max_depth=4)

modelo = modelo.fit(X, y)

fig, ax = plt.subplots(figsize=(20, 12)) #Tamaño del gráfico
tree.plot_tree(modelo, fontsize = 10)

num_pokemon.columns


#prediccion

from sklearn.model_selection import train_test_split #Separar el data set en training y test
from sklearn.metrics import accuracy_score #Métricas de la predicción del modelo. Precisión
from sklearn.metrics import confusion_matrix 

X = num_pokemon.values[:, :-1] #Variables explicativas
y = num_pokemon.values[:, -1] #variable objetivo
y = y.astype("int")

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

modelo = DecisionTreeClassifier(criterion='entropy', max_depth=4)

modelo = modelo.fit(X_train, y_train)

y_prediccion = modelo.predict(X_test)

accuracy_score(y_test, y_prediccion)

pd.DataFrame(
    confusion_matrix(y_test, y_prediccion),
    columns=['Predicted Not Class', 'Predicted Class'],
    index=['True Not Class', 'True Class']
)

fig, ax = plt.subplots(figsize=(20, 12)) #Tamaño del gráfico
tree.plot_tree(modelo, fontsize = 10)

#CLUSTERING
from sklearn.cluster import KMeans

K_pokemon = num_pokemon[["HP", "Attack", "Legendary"]] 

kmeans_pokemon = KMeans(n_clusters = 4)

#Ajustar modelo
kmeans_pokemon.fit(K_pokemon)

#Dibujar el clústering
centroids = kmeans_pokemon.cluster_centers_
labels = kmeans_pokemon.labels_

#Dibujar los puntos con los colores de cada clúster
colors = ["g.","r.","c.","y."]
for i in range(len(K_pokemon)):
    plt.plot(K_pokemon.iloc[i,0], K_pokemon.iloc[i,1], colors[labels[i]], markersize = 10)
    
#Dibujar los centroides de cada clúster
plt.scatter(centroids[:, 0],centroids[:, 1], marker = "x", s=150, linewidths = 5, zorder = 10)

plt.show()



















