import os
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
    return nuevo_dato
