import numpy as np

inputs = [1.0, 2.0, 3.0, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]


biases = 2
#always add weights before inputs
output = np.dot(weights, inputs) + biases

print(output)