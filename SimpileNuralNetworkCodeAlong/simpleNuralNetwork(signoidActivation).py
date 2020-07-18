import numpy as np


def signoid(x):
    return 1 / (1 + np.exp(-x))

def signoid_dirivitive(x):
    return x * (1 - x)


training_inputs = np.array([[0, 0, 1],
                            [1, 1, 1],
                            [1, 0, 1],
                            [0, 1, 1]])
training_outputs = np.array([[0, 1, 1, 0]]).T

np.random.seed(1)

synaptic_weights = 2 * np.random.random((3, 1)) - 1
print('Random starting synaptic weights')
print(synaptic_weights)

for iteration in range(100000):
    input_layer = training_inputs

    outputs = signoid(np.dot(input_layer, synaptic_weights))

    error = training_outputs - outputs

    ajustments = error * signoid_dirivitive(outputs)

    synaptic_weights += np.dot(input_layer.T, ajustments)

print('Synaptic weights after training')
print(synaptic_weights)

print('Outputs after training')
print(outputs)
