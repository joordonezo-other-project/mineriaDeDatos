from KNN import knn_clasificacion

def matriz_confusion(datos_entrenamiento, datos_prueba, k):
    #evaluar el modelo con los datos de prueba
    #matriz de confusión 
    m_c = [[0]*3 for i in range(3)]
    for dato in datos_prueba:
        predicciones = knn_clasificacion(dato[:-1], datos_entrenamiento, k)
        if dato[-1] == predicciones:
            #calcular la matriz de confusión
            if dato[-1] == "Iris-setosa":
                if dato[-1] == predicciones:
                    m_c[0][0] += 1
                else:
                    m_c[0][1] += 1
            elif dato[-1] == "Iris-versicolor":
                if dato[-1] == predicciones:
                    m_c[1][1] += 1
                else:
                    m_c[1][0] += 1
            elif dato[-1] == "Iris-virginica":
                if dato[-1] == predicciones:
                    m_c[2][2] += 1
                else:
                    m_c[2][0] += 1
        else:
            if dato[-1] == "Iris-setosa":
                m_c[0][2] += 1
            elif dato[-1] == "Iris-versicolor":
                m_c[1][2] += 1
            elif dato[-1] == "Iris-virginica":
                m_c[2][2] += 1
    #m_c es la matriz de confusión
    for x in m_c:
        print(x)
    print("Exactitud: ", (m_c[0][0]+m_c[1][1]+m_c[2][2])/len(datos_prueba)*100, "%")
    print("Precision: ", m_c[0][0]/(m_c[0][0]+m_c[0][1]+m_c[0][2])*100, "%")
    print("Sensibilidad: ", m_c[0][0]/(m_c[0][0]+m_c[1][0]+m_c[2][0])*100, "%")
    print("Especificidad: ", (m_c[1][1]+m_c[1][2]+m_c[2][1]+m_c[2][2])/(m_c[1][0]+m_c[1][1]+m_c[1][2]+m_c[2][0]+m_c[2][1]+m_c[2][2])*100, "%")
