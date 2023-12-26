import numpy as np

X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)
X = X / np.amax(X, axis=0)
y = y / 100


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def derivatives_sigmoid(x):
    return x * (1 - x)


num_epochs = 5
learning_rate = 0.1
inputlayer_neurons = 2
hiddenlayer_neurons = 3
output_neurons = 1

weights_hidden = np.random.uniform(size=(inputlayer_neurons, hiddenlayer_neurons))
biases_hidden = np.random.uniform(size=(1, hiddenlayer_neurons))
weights_output = np.random.uniform(size=(hiddenlayer_neurons, output_neurons))
biases_output = np.random.uniform(size=(1, output_neurons))

for epoch in range(num_epochs):
    hidden_layer_input = np.dot(X, weights_hidden) + biases_hidden
    hidden_layer_activation = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_activation, weights_output) + biases_output
    output = sigmoid(output_layer_input)
    output_error = y - output
    output_delta = output_error * derivatives_sigmoid(output)
    hidden_layer_error = output_delta.dot(weights_output.T)
    hidden_layer_delta = hidden_layer_error * derivatives_sigmoid(hidden_layer_activation)
    weights_output += hidden_layer_activation.T.dot(output_delta) * learning_rate
    weights_hidden += X.T.dot(hidden_layer_delta) * learning_rate
    print("Predicted Output: \n", output)

print("Input: \n" + str(X))
print("Actual Output: \n" + str(y))
