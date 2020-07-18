import numpy as np


class NuralNetwork():

    def __init__(self):
        np.random.seed(1)

        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivitive(self, x):
        return x * (1 - x)

    def train(self, training_inputs, training_ouputs, training_iterations):
        for iteration in range(training_iterations):
            output = self.think(training_inputs)
            error = training_ouputs - output
            ajustments = np.dot(training_inputs.T, error * self.sigmoid_derivitive(output))
            self.synaptic_weights += ajustments

    def think(self, inputs):
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))

        return output


if __name__ == "__main__":
    neural_network = NuralNetwork()
    print("Random synaptic weights: ")
    print(neural_network.synaptic_weights)

    training_inputs = np.array([[0, 0, 1],
                                [1, 1, 1],
                                [1, 0, 1],
                                [0, 1, 1]])

    training_outputs = np.array([[0, 1, 1, 0]]).T

    print("training")
    neural_network.train(training_inputs, training_outputs, 10000)

    print("synaptic weights after training: ")
    print(neural_network.synaptic_weights)

    a = str(input("Input 1: "))
    b = str(input("Input 2: "))
    c = str(input("Input 3: "))

    print("new dataset: input data = ", a, b, c)
    print("output Data:")
    print(neural_network.think(np.array([a, b, c])))
