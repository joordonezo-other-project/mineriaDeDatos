import csv
import random
import os
import numpy as np

# Fix random seed for reproducibility
random.seed(113)

#ruta al archivo CSV
relative_path = 'data/IRIS.csv'
absolute_path = os.path.dirname(__file__)
full_path = os.path.join(absolute_path, relative_path)

# Load dataset
with open(full_path) as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader, None)  # skip header
    dataset = list(csvreader)

# Change string value to numeric
class_mapping = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
for row in dataset:
    row[4] = class_mapping[row[4]]
    row[:4] = [float(row[j]) for j in range(len(row))]

# Split x and y (feature and target)
random.shuffle(dataset)
split_ratio = 0.8
split_index = int(len(dataset) * split_ratio)
datatrain = dataset[:split_index]
datatest = dataset[split_index:]
train_X = [data[:4] for data in datatrain]
train_y = [data[4] for data in datatrain]
test_X = [data[:4] for data in datatest]
test_y = [data[4] for data in datatest]

# Define MLP parameters
input_size = 4
hidden_size = 4
output_size = 3
learning_rate = 0.005
epochs = 400

# Initialize weights and biases
weights_input_hidden = np.random.uniform(-1, 1, (input_size, hidden_size))
bias_hidden = np.zeros(hidden_size)
weights_hidden_output = np.random.uniform(-1, 1, (hidden_size, output_size))
bias_output = np.zeros(output_size)

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Training loop
for epoch in range(epochs):
    total_error = 0
    for i in range(len(train_X)):
        # Forward propagation
        hidden_input = np.dot(train_X[i], weights_input_hidden) + bias_hidden
        hidden_output = sigmoid(hidden_input)
        output_input = np.dot(hidden_output, weights_hidden_output) + bias_output
        output = sigmoid(output_input)

        # Compute the error
        target = np.zeros(output_size)
        target[int(train_y[i])] = 1  # Convert train_y[i] to an integer
        error = target - output
        total_error += np.sum(error**2)

        # Backpropagation
        delta_output = error * sigmoid_derivative(output)
        error_hidden = delta_output.dot(weights_hidden_output.T)
        delta_hidden = error_hidden * sigmoid_derivative(hidden_output)

        # Update weights and biases
        weights_hidden_output += learning_rate * np.outer(hidden_output, delta_output)
        bias_output += learning_rate * delta_output
        weights_input_hidden += learning_rate * np.outer(train_X[i], delta_hidden)
        bias_hidden += learning_rate * delta_hidden

    avg_error = total_error / len(train_X)
    if epoch % 100 == 0:
        print(f'Epoch {epoch}: Average Error = {avg_error:.6f}')


# Testing
def predict(X):
    hidden_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)
    output_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    output = sigmoid(output_input)
    return np.argmax(output)

predictions = [predict(x) for x in test_X]

# Calculate accuracy
accuracy = sum(p == int(y) for p, y in zip(predictions, test_y)) / len(test_y) * 100
print(f'Accuracy: {accuracy:.2f}%')


pred=predict([5.7,2.8,4.1,1.3])
if (pred==0):
    print("Iris-setosa")
elif(pred==1):
    print("Iris-versicolor")
else:
    print("Iris-virginica")

    