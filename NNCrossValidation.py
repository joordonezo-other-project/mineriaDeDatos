import csv
import random
import numpy as np
from Data import full_path

# Función de activación sigmoide y su derivada


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivada de la función sigmoide


def derivada_sigmoid(x):
    return x * (1 - x)


pedict_mapping = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
dataFoldsFinal = []


def entrenarModelo(tamaño_entrada, tamaño_oculto, tamaño_salida, tasa_aprendizaje, épocas, k):
    # Fijar una semilla aleatoria para reproducibilidad
    random.seed(113)

    # Cargar el conjunto de datos
    with open(full_path) as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader, None)  # saltar la cabecera
        dataset = list(csvreader)

    # Cambiar el valor de cadena a numérico
    class_mapping = {"Iris-setosa": 0,
                     "Iris-versicolor": 1, "Iris-virginica": 2}
    for row in dataset:
        row[4] = class_mapping[row[4]]
        row[:4] = [float(row[j]) for j in range(len(row))]

    # Implementar validación cruzada con k-fold
    fold_size = len(dataset) // k

    precisión_promedio = 0

    for fold in range(k):
        # Dividir los datos en entrenamiento y prueba
        inicio_prueba = fold * fold_size
        fin_prueba = (fold + 1) * fold_size
        datos_prueba = dataset[inicio_prueba:fin_prueba]
        datos_entrenamiento = dataset[:inicio_prueba] + dataset[fin_prueba:]

        entrenamiento_X = [data[:4] for data in datos_entrenamiento]
        entrenamiento_y = [data[4] for data in datos_entrenamiento]
        prueba_X = [data[:4] for data in datos_prueba]
        prueba_y = [data[4] for data in datos_prueba]

        # Inicializar pesos y sesgos
        pesos_entrada_oculta = np.random.uniform(
            -1, 1, (tamaño_entrada, tamaño_oculto))
        sesgo_oculto = np.zeros(tamaño_oculto)
        pesos_oculta_salida = np.random.uniform(-1,
                                                1, (tamaño_oculto, tamaño_salida))
        sesgo_salida = np.zeros(tamaño_salida)
        dataFoldsFinal.append({})
        # Bucle de entrenamiento
        for época in range(épocas):
            error_total = 0
            for i in range(len(entrenamiento_X)):
                # Propagación hacia adelante
                entrada_oculta = np.dot(
                    entrenamiento_X[i], pesos_entrada_oculta) + sesgo_oculto
                salida_oculta = sigmoid(entrada_oculta)
                entrada_salida = np.dot(
                    salida_oculta, pesos_oculta_salida) + sesgo_salida
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
                pesos_oculta_salida += tasa_aprendizaje * \
                    np.outer(salida_oculta, delta_salida)
                sesgo_salida += tasa_aprendizaje * delta_salida
                pesos_entrada_oculta += tasa_aprendizaje * \
                    np.outer(entrenamiento_X[i], delta_oculto)
                sesgo_oculto += tasa_aprendizaje * delta_oculto

            # Calcular el error promedio de la época
            error_promedio = error_total / len(entrenamiento_X)
            if época % 100 == 0:
                print(f'Época {época}: Error Promedio = {error_promedio:.6f}')

            finallyWeights = {
                'pesos_entrada_oculta': pesos_entrada_oculta,
                'sesgo_oculto': sesgo_oculto,
                'pesos_oculta_salida': pesos_oculta_salida,
                'sesgo_salida': sesgo_salida
            }
            dataFoldsFinal[fold] = finallyWeights
        # Pruebas

        def test(X):
            entrada_oculta = np.dot(X, pesos_entrada_oculta) + sesgo_oculto
            salida_oculta = sigmoid(entrada_oculta)
            entrada_salida = np.dot(
                salida_oculta, pesos_oculta_salida) + sesgo_salida
            salida = sigmoid(entrada_salida)
            return np.argmax(salida)

        predicciones = [test(x) for x in prueba_X]

        # Calcular la precisión
        precisión_fold = sum(p == int(y) for p, y in zip(
            predicciones, prueba_y)) / len(prueba_y) * 100
        print(f'Precisión para el fold {fold + 1}: {precisión_fold:.2f}%')
        precisión_promedio += precisión_fold

    # Calcular la precisión promedio de todos los folds
    precisión_promedio /= k
    print(f'Precisión Promedio: {precisión_promedio:.2f}%')

# Pruebas


def predecir(X, k):
    entrada_oculta = np.dot(X, dataFoldsFinal[k].get(
        "pesos_entrada_oculta")) + dataFoldsFinal[k].get("sesgo_oculto")
    salida_oculta = sigmoid(entrada_oculta)
    entrada_salida = np.dot(salida_oculta, dataFoldsFinal[k].get(
        "pesos_oculta_salida")) + dataFoldsFinal[k].get("sesgo_salida")
    salida = sigmoid(entrada_salida)
    return pedict_mapping[np.argmax(salida)]
