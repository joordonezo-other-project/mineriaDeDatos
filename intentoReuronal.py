import numpy as np
import os

# Establecer la ruta al archivo CSV
relative_path = 'data/IRIS.csv'
absolute_path = os.path.dirname(__file__)
full_path = os.path.join(absolute_path, relative_path)

# Crear listas para almacenar los datos
data = []
labels = []

# Abrir el archivo CSV y leer los datos
with open(full_path) as file:
    lines = file.readlines()
    lines.pop(0)  # Eliminar la primera línea (encabezado)

lines=np.random.choice(lines, len(lines), replace=False) #mezclar los datos
# Recorrer las líneas del archivo CSV
for line in lines:
    line = line.strip().split(',')
    data.append([float(x) for x in line[:-1]])  # Las primeras cuatro columnas son características
    labels.append(line[-1])  # La última columna es la etiqueta

# Mapear las etiquetas a números (Iris-setosa: 0, Iris-versicolor: 1, Iris-virginica: 2)
label_map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
y = [label_map[label] for label in labels]

# Convertir las listas en matrices NumPy
X = np.array(data)
y = np.array(y).reshape(-1, 1)  # Asegurar que y sea bidimensional

# Aplicar preprocesamiento
X[:,:4] = X[:,:4] - X[:,:4].mean(axis=0)  # Centrar los datos
imax = np.concatenate((X.max(axis=0)*np.ones((1, 4)), np.abs(X.min(axis=0)*np.ones((1, 4)))), axis=0).max(axis=0)
X[:,:4] = X[:,:4] / imax[:4]  # Escalar los datos

print(X)

# Inicialización de pesos y bias
input_size = 4
hidden_size = 20
output_size = 3
weights_input_hidden = np.random.rand(input_size, hidden_size)
bias_hidden = np.zeros((1, hidden_size))
weights_hidden_output = np.random.rand(hidden_size, output_size)
bias_output = np.zeros((1, output_size))

# Función de activación (sigmoide)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivada de la función de activación
def sigmoid_derivative(x):
    return x * (1 - x)

# Tasa de aprendizaje
learning_rate = 0.001

# Entrenamiento de la red neuronal
epochs = 1000

for epoch in range(epochs):
    # Forward propagation
    hidden_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)
    
    output_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    output = sigmoid(output_input)
    
    # Cálculo del error
    error = y - output
    
    # Backpropagation
    d_output = error * sigmoid_derivative(output)
    
    error_hidden = d_output.dot(weights_hidden_output.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)
    
    # Actualización de pesos y bias
    weights_hidden_output += hidden_output.T.dot(d_output) * learning_rate
    bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    
    weights_input_hidden += X.T.dot(d_hidden) * learning_rate
    bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

# Predicción
new_input = np.array([[5,3.4,1.5,0.2]])  # Puedes ajustar esto según tus necesidades
hidden_layer = sigmoid(np.dot(new_input, weights_input_hidden) + bias_hidden)
predicted_output = sigmoid(np.dot(hidden_layer, weights_hidden_output) + bias_output)

print("Predicción:")
print(predicted_output)
# #decir el tipo de flor
# if predicted_output[0][0] > predicted_output[0][1] and predicted_output[0][0] > predicted_output[0][2]:
#     print("Iris-setosa")
# elif predicted_output[0][1] > predicted_output[0][0] and predicted_output[0][1] > predicted_output[0][2]:
#     print("Iris-versicolor")
# else:
#     print("Iris-virginica")

