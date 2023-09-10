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
    print(datos)
    #randomizar los datos
    random.shuffle(datos)
    
    #método de entrenamiento Holdout
    datos_entrenamiento = datos[:int(len(datos)*0.7)]
    datos_prueba = datos[int(len(datos)*0.7):]

    return datos_entrenamiento, datos_prueba

#funcion para predecir la clase de un nuevo dato
def predecir(nuevo_dato,datos_entrenamiento, k):
    distancias = []
    for dato in datos_entrenamiento:
        distancias.append([distancia_euclidiana(nuevo_dato, dato[:-1]), dato[-1]])
    distancias.sort()
    clases = {}
    for i in range(k):
        if distancias[i][1] in clases:
            clases[distancias[i][1]] += 1
        else:
            clases[distancias[i][1]] = 1

    clase = ""
    maximo = 0
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


#funcion principal
def main():
    datos_entrenamiento, datos_prueba = leer_archivo(ruta)
    k = 5
    #evaluar el modelo
    correctos = 0
    for dato in datos_prueba:
        if dato[-1] == predecir(dato[1:-1], datos_entrenamiento, k):
            correctos += 1
    print("El modelo tuvo un porcentaje de acierto de: ", correctos/len(datos_prueba)*100, "%")
    nuevo_dato = pedir_datos()
    print("El nuevo dato pertenece a la clase: ", predecir(nuevo_dato, datos_entrenamiento, k))

if __name__ == "__main__":
    main()

