import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

X = [[1.0, 2.0, 3.0, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

X,y = spiral_data(100, 3)


class LayerDense:
    def __init__(self, n_inputs, n_nurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_nurons)

        # first peram is the shape of matix therfore pas in values as tuple
        self.biases = np.zeros((1, n_nurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class ActivationReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class ActivationSignoid:
    def forward(self, inputs):
        self.output = 1 / (1 + np.exp(-inputs))




layer1 = LayerDense(2, 5)
activation1 = ActivationReLU()

layer1.forward(X)
#print(layer1.output)
activation1.forward(layer1.output)
print(activation1.output)


'''
activation functions(rectified linar unit activation
inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
output = []
for i in inputs:
    this is reclu activation function
    if i > 0:
        output.append(i)
    if i<= 0:
        output.append(0)
    
    # so is this
    output.append(max(0,i))
print(output)
    '''