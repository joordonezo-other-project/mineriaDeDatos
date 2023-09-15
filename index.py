from HoldoutValidation import metodo_holdout, dividir_datos
from Data import pedir_datos
from CrossValidation import metodo_k_fold
from KNN import knn_clasificacion
from KNN import dataLines

#funcion principal
def main():
    datos_entrenamiento, datos_prueba = dividir_datos(dataLines)
    nuevo_dato, ks = pedir_datos()
    k = ks.get("k")
    k=int(k)
    k_folds = ks.get("k-fold")
    k_folds=int(k_folds)
    metodo_k_fold(datos_entrenamiento, k_folds)
    metodo_holdout(datos_entrenamiento, datos_prueba, k)
    etiqueta_predicha = knn_clasificacion(nuevo_dato, dataLines, k)
    
    print(f'La etiqueta con k={k} predicha para el nuevo punto en KNN es: {etiqueta_predicha}')

    # print(f"El nuevo dato HoldoutValidation con k={k} pertenece a la clase: ", predecir_holdout(nuevo_dato, datos_entrenamiento, k))

if __name__ == "__main__":
    main()