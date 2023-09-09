import os
import math
relative_path = 'data/IRIS.csv'
absolute_path = os.path.dirname(__file__)
full_path = os.path.join(absolute_path, relative_path)


def splitByComma(item):
    return item.split(',')


def distancia_euclidiana(punto1, punto2):
    suma_cuadrados = sum((float(x) - float(y)) ** 2 for x,
                         y in zip(punto1, punto2))
    return math.sqrt(suma_cuadrados)


def knn_clasificacion(conjunto_entrenamiento, etiquetas_entrenamiento, nuevo_punto, k):
    distancias = []

    for i, punto in enumerate(conjunto_entrenamiento):
        distancia = distancia_euclidiana(punto, nuevo_punto)
        distancias.append((distancia, etiquetas_entrenamiento[i]))

    distancias.sort()
    k_vecinos = distancias[:k]

    contador = {}
    for _, etiqueta in k_vecinos:
        contador[etiqueta] = contador.get(etiqueta, 0) + 1

    etiqueta_predicha = max(contador, key=contador.get)
    return etiqueta_predicha


data = open(full_path, mode='r')
data = data.read()
dataLines = data.split('\n')
dataLines = map(splitByComma, dataLines)
dataLines = list(dataLines)
headers = dataLines.pop(0)
# print("cabeceras " + str(headers)+"\n")
# print(dataLines)

dataWithoutLastColumn = [fila[:-1] for fila in dataLines]
dataWithoutLastColumn = [fila for fila in dataWithoutLastColumn if fila]
etiquetas_entrenamiento = [fila[-1] for fila in dataLines if fila[-1]]
nuevo_registros_a_clasificar = [6, 2.2, 4, 1]
k = 2

etiqueta_predicha = knn_clasificacion(
    dataWithoutLastColumn, etiquetas_entrenamiento, nuevo_registros_a_clasificar, k)
print(f'La etiqueta predicha para el nuevo punto es: {etiqueta_predicha}')
