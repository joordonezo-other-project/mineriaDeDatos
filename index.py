from HoldoutValidation import leer_archivo
from HoldoutValidation import metodo_holdout
from Data import pedir_datos
from HoldoutValidation import predecir_holdout
from CrossValidation import metodo_k_fold
from CrossValidation import predecir_cross
from KNN import knn_clasificacion
from KNN import dataWithoutLastColumn
from KNN import etiquetas_entrenamiento
from Data import full_path
#funcion principal
def main():
    datos_entrenamiento, datos_prueba = leer_archivo(full_path)
    nuevo_dato = pedir_datos()
    k =10

    metodo_k_fold(datos_entrenamiento, k)
    metodo_holdout(datos_entrenamiento, datos_prueba, k)
    etiqueta_predicha = knn_clasificacion(dataWithoutLastColumn, etiquetas_entrenamiento, nuevo_dato, k)
    
    print(f'La etiqueta con k={k} predicha para el nuevo punto en KNN es: {etiqueta_predicha}')

    print(f"El nuevo dato CrossValidation con k={k} pertenece a la clase: ", predecir_cross(nuevo_dato, datos_entrenamiento, k))
    
    print(f"El nuevo dato HoldoutValidation con k={k} pertenece a la clase: ", predecir_holdout(nuevo_dato, datos_entrenamiento, k))

if __name__ == "__main__":
    main()