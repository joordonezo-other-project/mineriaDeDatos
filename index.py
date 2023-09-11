from HoldoutValidation import leer_archivo
from HoldoutValidation import metodo_holdout
from HoldoutValidation import pedir_datos
from HoldoutValidation import predecir
from Data import full_path
#funcion principal
def main():
    datos_entrenamiento, datos_prueba = leer_archivo(full_path)
    k =10
    #metodo_k_fold(datos_entrenamiento, k)
    metodo_holdout(datos_entrenamiento, datos_prueba, k)
    nuevo_dato = pedir_datos()
    print("El nuevo dato pertenece a la clase: ", predecir(nuevo_dato, datos_entrenamiento, k))

if __name__ == "__main__":
    main()