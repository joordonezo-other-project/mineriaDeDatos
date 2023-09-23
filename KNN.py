from Data import full_path, distancia_euclidiana
import random
def splitByComma(item):
    return item.split(',')

data = open(full_path, mode='r')
data = data.read()
dataLines = data.split('\n')
dataLines = map(splitByComma, dataLines)
dataLines = list(dataLines)
headers = dataLines.pop(0)
dataLines= [fila for fila in dataLines if len(fila)>1]
random.shuffle(dataLines)

#funcion para predecir la clase de un nuevo dato
def knn_clasificacion(nuevo_dato,datos, k):
    distancias = []
    #calcular la distancia euclidiana entre el nuevo dato y los datos de entrenamiento y guardarlos en una lista
    for dato in datos:
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



