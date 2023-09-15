import random
from KNN import knn_clasificacion 

#funcion para leer el archivo y dividir los datos en entrenamiento y prueba
def dividir_datos(datos):
    #randomizar los datos
    random.shuffle(datos)
    #método de entrenamiento Holdout
    porcentaje_datos = 0.7
    datos_entrenamiento = datos[:int(len(datos)*porcentaje_datos)]
    datos_prueba = datos[int(len(datos)*porcentaje_datos):]
    return datos_entrenamiento, datos_prueba

def metodo_holdout(datos_entrenamiento, datos_prueba, k):
    #evaluar el modelo con los datos de prueba
    correctos = 0
    for dato in datos_prueba:
        predicciones = knn_clasificacion(dato[:-1], datos_entrenamiento, k)
        if dato[-1] == predicciones:
            correctos += 1
    print("HOLDOUT - El modelo tuvo un porcentaje de acierto de: ", correctos/len(datos_prueba)*100, "%")

