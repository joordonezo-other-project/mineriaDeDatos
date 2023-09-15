import os
import math
relative_path = 'data/IRIS.csv'
absolute_path = os.path.dirname(__file__)
full_path = os.path.join(absolute_path, relative_path)

#funcion para pedir los datos de un nuevo dato
def pedir_datos():
    nuevo_dato = []
    nuevo_dato.append(input("Sepal length: "))
    nuevo_dato.append(input("Sepal width: "))
    nuevo_dato.append(input("Petal length: "))
    nuevo_dato.append(input("Petal width: "))
    ks = {
        "k" : input("Ingrese el valor de k: "),
        "k-fold" : input("Ingrese cantidad de grupos k-folds: ")
    }
    return nuevo_dato, ks

#funcion para calcular la distancia euclidiana
def distancia_euclidiana(punto1, punto2):
    suma = 0
    for i in range(len(punto1)):
        suma += (float(punto1[i]) - float(punto2[i]))**2
    return math.sqrt(suma)
