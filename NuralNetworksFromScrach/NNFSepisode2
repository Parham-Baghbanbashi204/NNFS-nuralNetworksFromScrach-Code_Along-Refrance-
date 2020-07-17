'''
Creates a simple layer of neurons, with 4 inputs.
Associated YT NNFS tutorial: https://www.youtube.com/watch?v=lGLto9Xd7bU
'''
'''method one of makeing a layer of nurons network
inputs = [1.0, 2.0, 3.0, 2.5]

weights1 = [0.2, 0.8, -0.5, 1.0]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]

bias1 = 2.0
bias2 = 3.0
bias3 = 0.5

output = [inputs[0]*weights1[0] + inputs[1]*weights1[1] + inputs[2]*weights1[2] + inputs[3]*weights1[3] + bias1,
          inputs[0]*weights2[0] + inputs[1]*weights2[1] + inputs[2]*weights2[2] + inputs[3]*weights2[3] + bias2,
          inputs[0]*weights3[0] + inputs[1]*weights3[1] + inputs[2]*weights3[2] + inputs[3]*weights3[3] + bias3]
print(output)
'''
#method of simplifying one nuron calculation
inputs = [1.0, 2.0, 3.0, 2.5]
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2.0, 3.0, 0.5]

Outputs = []
for nuron_weights, nuron_biases in zip(weights, biases):
    nuron_output = 0
    for nuron_input, weight in zip(inputs, nuron_weights):
        nuron_output += nuron_input * weight
    nuron_output += nuron_biases
    Outputs.append(nuron_output)
print(Outputs)
