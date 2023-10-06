import csv
import random
import os
import numpy as np

# Fijar una semilla aleatoria para reproducibilidad
random.seed(113)

# Ruta al archivo CSV
ruta_relativa = 'data/IRIS.csv'
ruta_absoluta = os.path.dirname(__file__)
ruta_completa = os.path.join(ruta_absoluta, ruta_relativa)

# Cargar el conjunto de datos
with open(ruta_completa) as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader, None)  # saltar la cabecera
    dataset = list(csvreader)

# Cambiar el valor de cadena a numérico
class_mapping = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
for row in dataset:
    row[4] = class_mapping[row[4]]
    row[:4] = [float(row[j]) for j in range(len(row))]

# Dividir x e y (características y objetivos)
random.shuffle(dataset)
ratio_division = 0.8
indice_division = int(len(dataset) * ratio_division)
datos_entrenamiento = dataset[:indice_division]
datos_prueba = dataset[indice_division:]
entrenamiento_X = [data[:4] for data in datos_entrenamiento]
entrenamiento_y = [data[4] for data in datos_entrenamiento]
prueba_X = [data[:4] for data in datos_prueba]
prueba_y = [data[4] for data in datos_prueba]

# Definir los parámetros de la red neuronal MLP
tamaño_entrada = 4
tamaño_oculto = 8
tamaño_salida = 3
tasa_aprendizaje = 0.005
épocas = 1000

# Inicializar pesos y sesgos
pesos_entrada_oculta = np.random.uniform(-1, 1, (tamaño_entrada, tamaño_oculto))
sesgo_oculto = np.zeros(tamaño_oculto)
pesos_oculta_salida = np.random.uniform(-1, 1, (tamaño_oculto, tamaño_salida))
sesgo_salida = np.zeros(tamaño_salida)

# Función de activación sigmoide y su derivada
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivada de la función sigmoide
def derivada_sigmoid(x):
    return x * (1 - x)

# Bucle de entrenamiento
for época in range(épocas):
    error_total = 0
    for i in range(len(entrenamiento_X)):
        # Propagación hacia adelante
        entrada_oculta = np.dot(entrenamiento_X[i], pesos_entrada_oculta) + sesgo_oculto
        salida_oculta = sigmoid(entrada_oculta)
        entrada_salida = np.dot(salida_oculta, pesos_oculta_salida) + sesgo_salida
        salida = sigmoid(entrada_salida)

        # Calcular el error
        objetivo = np.zeros(tamaño_salida)
        objetivo[int(entrenamiento_y[i])] = 1
        error = objetivo - salida
        error_total += np.sum(error**2)

        # Retropropagación
        delta_salida = error * derivada_sigmoid(salida)
        error_oculto = delta_salida.dot(pesos_oculta_salida.T)
        delta_oculto = error_oculto * derivada_sigmoid(salida_oculta)

        # Actualizar pesos y sesgos
        pesos_oculta_salida += tasa_aprendizaje * np.outer(salida_oculta, delta_salida)
        sesgo_salida += tasa_aprendizaje * delta_salida
        pesos_entrada_oculta += tasa_aprendizaje * np.outer(entrenamiento_X[i], delta_oculto)
        sesgo_oculto += tasa_aprendizaje * delta_oculto

    # Calcular el error promedio de la época
    error_promedio = error_total / len(entrenamiento_X)
    if época % 100 == 0:
        print(f'Época {época}: Error Promedio = {error_promedio:.6f}')

# Pruebas
def predecir(X):
    entrada_oculta = np.dot(X, pesos_entrada_oculta) + sesgo_oculto
    salida_oculta = sigmoid(entrada_oculta)
    entrada_salida = np.dot(salida_oculta, pesos_oculta_salida) + sesgo_salida
    salida = sigmoid(entrada_salida)
    return np.argmax(salida)

predicciones = [predecir(x) for x in prueba_X]

# Calcular la precisión
precisión = sum(p == int(y) for p, y in zip(predicciones, prueba_y)) / len(prueba_y) * 100
print(f'Precisión: {precisión:.2f}%')


# Predicción de un nuevo dato
def pedir_datos():
    nuevo_dato = []
    nuevo_dato.append(float(input("Longitud del Sépalo: ")))
    nuevo_dato.append(float(input("Ancho del Sépalo: ")))
    nuevo_dato.append(float(input("Longitud del Pétalo: ")))
    nuevo_dato.append(float(input("Ancho del Pétalo: ")))
    return nuevo_dato

nuevo_dato = pedir_datos()
mapeo_clases = {0: "Iris-Setosa", 1: "Iris-Versicolor", 2: "Iris-Virginica"}
etiqueta_predicha = predecir(nuevo_dato)
print(mapeo_clases[etiqueta_predicha])


    