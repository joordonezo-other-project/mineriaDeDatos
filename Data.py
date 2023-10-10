import os
import math
relative_path = 'data/IRIS.csv'
absolute_path = os.path.dirname(__file__)
full_path = os.path.join(absolute_path, relative_path)

# funcion para pedir los datos de un nuevo dato


def pedir_datos():
    nuevo_dato = []
    nuevo_dato.append(float(input("Sepal length: ")))
    nuevo_dato.append(float(input("Sepal width: ")))
    nuevo_dato.append(float(input("Petal length: ")))
    nuevo_dato.append(float(input("Petal width: ")))
    ks = {
        "k": int(input("Ingrese el valor de k: ")),
        "k-fold": int(input("Ingrese cantidad de grupos k-folds: "))
    }
    return nuevo_dato, ks

def pedir_datos_conk():
    nuevo_dato = []
    nuevo_dato.append(float(input("Sepal length: ")))
    nuevo_dato.append(float(input("Sepal width: ")))
    nuevo_dato.append(float(input("Petal length: ")))
    nuevo_dato.append(float(input("Petal width: ")))
    return nuevo_dato
# funcion para calcular la distancia euclidiana


def distancia_euclidiana(punto1, punto2):
    suma = 0
    for i in range(len(punto1)):
        suma += (float(punto1[i]) - float(punto2[i]))**2
    return math.sqrt(suma)


def obtener_parametros():
    tam_oculto = int(
        input("Ingrese el numero de neuronas de la capa oculta: "))
    taza_aprendizaje = float(input("Ingrese la taza de aprendizaje: "))
    epocas = int(input("Ingrese número de epocas: "))
    k_folds = int(input("Ingrese número k-folds: "))
    return tam_oculto, taza_aprendizaje, epocas, k_folds
