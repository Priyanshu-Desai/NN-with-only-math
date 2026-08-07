import math

# Simple neural network implemented using only math and basic data structures.
# Provides a lightweight Matrix helper class, simple layer types, and a
# `NeuralNetwork` implementation with forward and backward passes for
# educational and demonstration purposes.

# Matrix maths: lightweight matrix class providing basic operations used
# throughout the neural network (addition, subtraction, matmul, elementwise).
class Matrix:
    def __init__(self, dim, *nums):
        rows, cols = dim.split('x')
        rows, cols = int(rows), int(cols)
        self.dim = (rows, cols)

        self.matrix = [[0 for _ in range(cols)] for _ in range(rows)]

        iter_nums = iter(nums)
        for i in range(rows):
            for j in range(cols):
                try:
                    self.matrix[i][j] = float(next(iter_nums))
                except StopIteration as exc:
                    raise StopIteration(f"Error: Not enough numbers provided to fill the matrix of dimension {dim}. Remaining elements will be set to 0.") from exc
                except ValueError as exc:
                    raise ValueError("Error: Enter a number as an argument. Remaining elements will be set to 0.") from exc
                except Exception as e:
                    raise e
    
    def __add__(self, other):
        if self.dim != other.dim:
            raise ValueError("Matrices must have the same dimensions for addition.")
        resValues = []
        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                resValues.append(self.matrix[i][j] + other.matrix[i][j])
        result = Matrix(f"{self.dim[0]}x{self.dim[1]}", *resValues)
        return result

    def __sub__(self, other):
        if self.dim != other.dim:
            raise ValueError("Matrices must have the same dimensions for subtraction.")
        resValues = []
        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                resValues.append(self.matrix[i][j] - other.matrix[i][j])
        result = Matrix(f"{self.dim[0]}x{self.dim[1]}", *resValues)
        return result

    def __matmul__(self, other):
        if self.dim[1] != other.dim[0]:
            raise ValueError("Number of columns in the first matrix must be equal to the number of rows in the second matrix for multiplication.")
        resValues = []
        for i in range(self.dim[0]):
            for j in range(other.dim[1]):
                row_to_mul = self.matrix[i]
                col_to_mul = [other.matrix[k][j] for k in range(other.dim[0])]
                resValues.append(sum(a * b for a, b in zip(row_to_mul, col_to_mul)))
        result = Matrix(f"{self.dim[0]}x{other.dim[1]}", *resValues)
        return result

    def __mul__(self, other):
        if self.dim != other.dim:
            raise ValueError("Matrices must have the same dimensions for Hadamard product.")
        resValues = []
        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                resValues.append(self.matrix[i][j] * other.matrix[i][j])
        result = Matrix(f"{self.dim[0]}x{self.dim[1]}", *resValues)
        return result

    def show(self):
        for row in self.matrix:
            print(row) 

    def flatten(self):
        return [item for sublist in self.matrix for item in sublist]

    def transpose(self):
        transposed_values = []
        for j in range(self.dim[1]):
            for i in range(self.dim[0]):
                transposed_values.append(self.matrix[i][j])
        return Matrix(f"{self.dim[1]}x{self.dim[0]}", *transposed_values)   

    def scalar_multiply(self, scalar):
        resValues = []
        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                resValues.append(self.matrix[i][j] * scalar)
        result = Matrix(f"{self.dim[0]}x{self.dim[1]}", *resValues)
        return result

# nn layers
    # Neural network layer types
    # - `InputLayer`: holds input neuron values (stored in `neuronBiases` as a column vector)
    # - `MiddleLayer`: hidden layer with an activation function (sigmoid) and biases
    # - `OutputLayer`: final layer with `softmax` activation and biases
    # - `WeightLayer`: stores the weights matrix connecting two neuron layers
class InputLayer():
    def __init__(self, neurons):
        self.neuronBiases = Matrix(f"{neurons}x1", *[0 for _ in range(neurons)])

class MiddleLayer():
    def __init__(self, neurons, *biases):
        self.activationFunction = sigmoid
        self.neuronBiases = Matrix(f"{neurons}x1", *biases)

class OutputLayer():
    def __init__(self, neurons, *biases):
        self.activationFunction = softmax
        self.neuronBiases = Matrix(f"{neurons}x1", *biases)

class WeightLayer:
    def __init__(self, rows, cols, *weights):
        self.weights = Matrix(f"{rows}x{cols}", *weights)

# The `NeuralNetwork` class builds a network from a layer configuration
# and provides methods to set inputs, run a forward pass, compute loss,
# backpropagate errors, compute gradients, update weights, and serialize
# the network to a simple dictionary structure.
class NeuralNetwork:
    # JSON input
    # {
    #     1: {
    #         type: 'input',
    #         size: 3
    #     }
    #     2: {
    #         type: 'weights',
    #         size: [3, 4],
    #         weights: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    #     }
    #     3: {
    #         type: 'output',
    #         size: 4,
    #         biases: [0, 0, 0, 0]
    #     }
    # }
    def __init__(self, layerConfig):
        self.neuronLayers = []
        self.weightLayers = [None]

        if isinstance(layerConfig, dict):
            sorted_keys = sorted(layerConfig.keys(), key=int)
            layerConfig = [layerConfig[key] for key in sorted_keys]

        for layer in layerConfig:
            layerType = layer['type']
            layerSize = layer['size'] if 'size' in layer else None
            layerBiases = layer['biases'] if 'biases' in layer else None
            layerWeights = layer['weights'] if 'weights' in layer else None
            if layerType == 'input':
                self.neuronLayers.append(InputLayer(layerSize))
            elif layerType == 'middle':
                self.neuronLayers.append(MiddleLayer(layerSize, *layerBiases))
            elif layerType == 'output':
                self.neuronLayers.append(OutputLayer(layerSize, *layerBiases))
            elif layerType == 'weight':
                self.weightLayers.append(WeightLayer(layerSize[0], layerSize[1], *layerWeights))
        self.layerOutputs = []
        self.delta_values = []


    def setInput(self, *inputValues):
        if len(inputValues) != self.neuronLayers[0].neuronBiases.dim[0]:
            raise ValueError("Input values must match the number of neurons in the input layer.")
        self.neuronLayers[0].neuronBiases = Matrix(f"{len(inputValues)}x1", *inputValues)
        self.layerOutputs = [self.neuronLayers[0].neuronBiases]

    def forward(self):
        self.layerOutputs = [self.neuronLayers[0].neuronBiases]
        nextInput = self.neuronLayers[0].neuronBiases
        for i in range(1, len(self.neuronLayers)):
            nextInput = self.neuronLayers[i].activationFunction(self.weightLayers[i].weights @ nextInput + self.neuronLayers[i].neuronBiases)
            self.layerOutputs.append(nextInput)
        return (nextInput, self.layerOutputs)

    def loss(self, prediction, target):
        if target < 0 or target >= prediction.dim[0]:
            raise ValueError("Target index is out of bounds for the prediction matrix.")
        return -1 * math.log(prediction.matrix[target][0])
    
    def backpropogate(self, prediction, target):
        self.delta_values  = []
        for layer in reversed(range(1, len(self.neuronLayers))):
            if isinstance(self.neuronLayers[layer], OutputLayer):
                self.delta_values.append(prediction - Matrix(f"{prediction.dim[0]}x1", *[1 if i == target else 0 for i in range(prediction.dim[0])]))
            elif isinstance(self.neuronLayers[layer], MiddleLayer):
                # find the delta value for a middle layer that uses sigmoid activation function
                prev_delta = self.delta_values[-1]
                prev_weights = self.weightLayers[layer + 1].weights
                layer_output = self.layerOutputs[layer]
                delta = (prev_weights.transpose() @ prev_delta) * sigmoid_derivative(layer_output)
                self.delta_values.append(delta)
        self.delta_values.reverse()
        return self.delta_values

    def gradients(self):
        weight_gradients = []
        bias_gradients = self.delta_values
        for layer in range(1, len(self.neuronLayers)):
            gradient = self.delta_values[layer - 1] @ self.layerOutputs[layer - 1].transpose()
            weight_gradients.append(gradient)
        return (weight_gradients, bias_gradients)

    def update(self, weight_gradients, bias_gradients, learning_rate):
        for layer in range(1, len(self.neuronLayers)):
            self.weightLayers[layer].weights = self.weightLayers[layer].weights - weight_gradients[layer - 1].scalar_multiply(learning_rate)
            self.neuronLayers[layer].neuronBiases = self.neuronLayers[layer].neuronBiases - bias_gradients[layer - 1].scalar_multiply(learning_rate)

    def save(self):
        data = {}
        for i, layer in enumerate(self.neuronLayers):
            if isinstance(layer, InputLayer):
                data[2*i] = {
                    'type': 'input',
                    'size': layer.neuronBiases.dim[0],
                    'biases': layer.neuronBiases.flatten()
                }
            elif isinstance(layer, MiddleLayer):
                data[2*i] = {
                    'type': 'middle',
                    'size': layer.neuronBiases.dim[0],
                    'biases': layer.neuronBiases.flatten()
                }
            elif isinstance(layer, OutputLayer):
                data[2*i] = {
                    'type': 'output',
                    'size': layer.neuronBiases.dim[0],
                    'biases': layer.neuronBiases.flatten()
                }
        for i, weight_layer in enumerate(self.weightLayers[1:], start=1):
            data[2*i-1] = {
                'type': 'weight',
                'size': [weight_layer.weights.dim[0], weight_layer.weights.dim[1]],
                'weights': weight_layer.weights.flatten()
            }
        return data


# Activation functions and derivatives used by layers:
# `sigmoid` - logistic activation applied element-wise.
def sigmoid(arr):
    arr = arr.flatten()
    return Matrix(f"{len(arr)}x1", *[1/(1+math.exp(-x)) for x in arr])

# `softmax` - converts raw scores into probabilities that sum to 1.
def softmax(arr):
    arr = arr.flatten()
    exp = [math.exp(x) for x in arr]
    sum_exp = sum(exp)
    return Matrix(f"{len(arr)}x1", *[x / sum_exp for x in exp])

# `sigmoid_derivative` - derivative of the sigmoid given sigmoid outputs.
def sigmoid_derivative(arr):
    arr = arr.flatten()
    return Matrix(f"{len(arr)}x1", *[x * (1 - x) for x in arr])
