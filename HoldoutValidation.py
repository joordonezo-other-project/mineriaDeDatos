import os
import math
import random

relative_path = 'data/IRIS.csv'
absolute_path = os.path.dirname(__file__)
ruta = os.path.join(absolute_path, relative_path)


#funcion para leer el archivo y dividir los datos en entrenamiento y prueba
def leer_archivo(ruta):
    archivo = open(ruta, "r")
    lineas = archivo.readlines()
    archivo.close()
    datos = []
    for linea in lineas:
        datos.append(linea.strip().split(","))
    datos.pop(0)

    #randomizar los datos
    random.shuffle(datos)
    
    #método de entrenamiento Holdout
    datos_entrenamiento = datos[:int(len(datos)*0.7)]
    datos_prueba = datos[int(len(datos)*0.7):]

    return datos_entrenamiento, datos_prueba

#funcion para predecir la clase de un nuevo dato
def predecir(nuevo_dato,datos_entrenamiento, k):
    distancias = []
    #calcular la distancia euclidiana entre el nuevo dato y los datos de entrenamiento y guardarlos en una lista
    for dato in datos_entrenamiento:
        distancias.append([distancia_euclidiana(nuevo_dato, dato[:-1]), dato[-1]])
    #ordenar la lista de distancias de menor a mayor, para saber cuales son los k vecinos mas cercanos. los que tengan menor distancia son los mas cercanos
    distancias.sort()
    clases = {}
    #calcular la clase del nuevo dato con los k vecinos mas cercanos de la clase
    for i in range(k):
        if distancias[i][1] in clases:
            clases[distancias[i][1]] += 1
        else:
            #si la clase no esta en el diccionario, se agrega y se le asigna el valor de 1
            clases[distancias[i][1]] = 1
    clase = ""
    maximo = 0
    #encontrar la clase con mas vecinos cercanos y retornar el que más se repite
    for key in clases:
        if clases[key] > maximo:
            maximo = clases[key]
            clase = key
    return clase

#funcion para calcular la distancia euclidiana
def distancia_euclidiana(punto1, punto2):
    suma = 0
    for i in range(len(punto1)):
        suma += (float(punto1[i]) - float(punto2[i]))**2
    return math.sqrt(suma)

#funcion para pedir los datos de un nuevo dato
def pedir_datos():
    nuevo_dato = []
    nuevo_dato.append(input("Sepal length: "))
    nuevo_dato.append(input("Sepal width: "))
    nuevo_dato.append(input("Petal length: "))
    nuevo_dato.append(input("Petal width: "))
    return nuevo_dato


# def metodo_k_fold(datos, k):
    # #dividir los datos en k grupos
    # grupos = []
    # for i in range(k):
    #     grupos.append([])
    # for i in range(len(datos)):
    #     grupos[i%k].append(datos[i])
    # #evaluar el modelo con los k grupos
    # for i in range(k):
    #     datos_entrenamiento = []
    #     datos_prueba = grupos[i]
    #     for j in range(k):
    #         if j != i:
    #             datos_entrenamiento += grupos[j]
    #     #evaluar el modelo con los datos de prueba
    #     correctos = 0
    #     for dato in datos_prueba:
    #         if dato[-1] == predecir(dato[:-1], datos_entrenamiento, k):
    #             correctos += 1
    #     print("El modelo tuvo un porcentaje de acierto de: ", correctos/len(datos_prueba)*100, "%")

def metodo_holdout(datos_entrenamiento, datos_prueba, k):
    #evaluar el modelo con los datos de prueba
    correctos = 0
    for dato in datos_prueba:
        if dato[-1] == predecir(dato[:-1], datos_entrenamiento, k):
            correctos += 1
    print("El modelo tuvo un porcentaje de acierto de: ", correctos/len(datos_prueba)*100, "%")



