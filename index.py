from HoldoutValidation import metodo_holdout, dividir_datos
from Data import pedir_datos, obtener_parametros, pedir_datos_conk
from CrossValidation import metodo_k_fold
from KNN import knn_clasificacion, dataLines
from MatrizConfusion import matriz_confusion
from NNCrossValidation import entrenarModelo, predecir


# funcion principal
def main():
    """4
    datos_entrenamiento, datos_prueba = dividir_datos(dataLines)
    nuevo_dato, ks = pedir_datos()
    k = ks.get("k")
    k = int(k)
    k_folds = ks.get("k-fold")
    k_folds = int(k_folds)

    metodo_k_fold(dataLines, k_folds,k)
    metodo_holdout(datos_entrenamiento, datos_prueba, k)
    matriz_confusion(datos_entrenamiento, datos_prueba, k)
    etiqueta_predicha = knn_clasificacion(nuevo_dato, dataLines, k)
    print(f'La etiqueta con k={k} predicha para el nuevo punto en KNN es: {etiqueta_predicha}')
    """
    tam_oculto, taza_aprendizaje, epocas, k_folds = obtener_parametros()
    finallyWeights = entrenarModelo(
        4, tam_oculto, 3, taza_aprendizaje, epocas, k_folds)

    while True:
        dato_predecir = pedir_datos_conk()
        prediccion = predecir(dato_predecir, pesos=finallyWeights)
        print(
            f'La etiqueta con k_folds={k_folds} predicha para el nuevo punto  es: {prediccion}')


if __name__ == "__main__":
    main()
