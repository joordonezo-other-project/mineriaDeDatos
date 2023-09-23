from HoldoutValidation import metodo_holdout, dividir_datos
from Data import pedir_datos
from CrossValidation import metodo_k_fold
from KNN import knn_clasificacion, dataLines
from MatrizConfusion import matriz_confusion


#funcion principal
def main():
    datos_entrenamiento, datos_prueba = dividir_datos(dataLines)
    nuevo_dato, ks = pedir_datos()
    k = ks.get("k")
    k=int(k)
    k_folds = ks.get("k-fold")
    k_folds=int(k_folds)
    metodo_k_fold(dataLines, k_folds,k)
    metodo_holdout(datos_entrenamiento, datos_prueba, k)
    matriz_confusion(datos_entrenamiento, datos_prueba, k)
    etiqueta_predicha = knn_clasificacion(nuevo_dato, dataLines, k)
    print(f'La etiqueta con k={k} predicha para el nuevo punto en KNN es: {etiqueta_predicha}')
if __name__ == "__main__":
    main()