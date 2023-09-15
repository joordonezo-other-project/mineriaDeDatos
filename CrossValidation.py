from KNN import knn_clasificacion 

def metodo_k_fold(datos, k, k_vecinos):
    #dividir los datos en k grupos
    grupos = []
    currency = []
    for i in range(k):
        grupos.append([])
    for i in range(len(datos)):
        grupos[i%k].append(datos[i])
    #evaluar el modelo con los k grupos
    for i in range(k):
        datos_entrenamiento = []
        datos_prueba = grupos[i]
        for j in range(k):
            if j != i:
                datos_entrenamiento += grupos[j]
        #evaluar el modelo con los datos de prueba
        correctos = 0
        for dato in datos_prueba:
            if dato[-1] == knn_clasificacion(dato[:-1], datos_entrenamiento, k_vecinos):
                correctos += 1
        currency.append(correctos/len(datos_prueba)*100)
    print("CROSS VALIDATION - El modelo tuvo un porcentaje de acierto de: ", sum(currency)/len(currency), "%")

