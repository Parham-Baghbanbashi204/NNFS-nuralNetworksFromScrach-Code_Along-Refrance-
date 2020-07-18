import numpy as np
inputs = [1.0, 2.0, 3.0, 2.5]
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2.0, 3.0, 0.5]

#add weights first becuse it is a matrix not a vector(indexed as array plus doesent throw shape errors
output = np.dot(weights, inputs) + biases
#returns an array when a matirx is involved

print(output)