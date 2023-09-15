import math
import random
from KNN import knn_clasificacion 

#funcion para leer el archivo y dividir los datos en entrenamiento y prueba
def dividir_datos(datos):
    #randomizar los datos
    random.shuffle(datos)
    #método de entrenamiento Holdout
    datos_entrenamiento = datos[:int(len(datos)*0.7)]
    datos_prueba = datos[int(len(datos)*0.7):]
    return datos_entrenamiento, datos_prueba

def metodo_holdout(datos_entrenamiento, datos_prueba, k):
    #evaluar el modelo con los datos de prueba
    correctos = 0
    for dato in datos_prueba:
        if dato[-1] == knn_clasificacion(dato[:-1], datos_entrenamiento, k):
            correctos += 1
    print("HOLDOUT - El modelo tuvo un porcentaje de acierto de: ", correctos/len(datos_prueba)*100, "%")

