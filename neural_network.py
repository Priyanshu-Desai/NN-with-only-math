import math
import random


# matrix maths
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

    def __mul__(self, other):
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

    def show(self):
        for row in self.matrix:
            print(row) 

    def flatten(self):
        if self.dim[1] != 1:
            raise ValueError("Matrix must be a column vector (n x 1) to flatten.")
        return [item for sublist in self.matrix for item in sublist]

    def transpose(self):
        transposed_values = []
        for j in range(self.dim[1]):
            for i in range(self.dim[0]):
                transposed_values.append(self.matrix[i][j])
        return Matrix(f"{self.dim[1]}x{self.dim[0]}", *transposed_values)

# nn layers
class Layer:
    def __init__(self, neurons):
        self.neuronWeights = Matrix(f"{neurons}x1", *[random.uniform(-1, 1) for _ in range(neurons)])

# nn layer types
class InputLayer(Layer):
    def __init__(self, neurons):
        super().__init__(neurons)

class MiddleLayer(Layer):
    def __init__(self, neurons):
        super().__init__(neurons)
        self.activationFunction = sigmoid
        self.neuronBiases = Matrix(f"{neurons}x1", *[random.uniform(-1, 1) for _ in range(neurons)])

class OutputLayer(Layer):
    def __init__(self, neurons):
        super().__init__(neurons)
        self.activationFunction = softmax
        self.neuronBiases = Matrix(f"{neurons}x1", *[random.uniform(-1, 1) for _ in range(neurons)])

class WeightLayer:
    def __init__(self, inputLayer, outputLayer):
        self.weights = Matrix(f"{outputLayer.neuronWeights.dim[0]}x{inputLayer.neuronWeights.dim[0]}", *[random.uniform(-1, 1) for _ in range(outputLayer.neuronWeights.dim[0] * inputLayer.neuronWeights.dim[0])])

# the nn
class NeuralNetwork:
    def __init__(self, *layers):
        self.neuronLayers = []
        self.weightLayers = [None]
        for layer in layers:
            if layer[0] == 'input':
                self.neuronLayers.append(InputLayer(layer[1]))
            elif layer[0] == 'middle':
                self.weightLayers.append(WeightLayer(self.neuronLayers[-1], MiddleLayer(layer[1])))
                self.neuronLayers.append(MiddleLayer(layer[1]))
            elif layer[0] == 'output':
                self.weightLayers.append(WeightLayer(self.neuronLayers[-1], OutputLayer(layer[1])))
                self.neuronLayers.append(OutputLayer(layer[1]))
        self.layerOutputs = []
    def forward(self):
        nextInput = self.neuronLayers[0].neuronWeights
        for i in range(1, len(self.neuronLayers)):
            nextInput = self.neuronLayers[i].activationFunction(self.weightLayers[i] * nextInput + self.neuronLayers[i].neuronBiases)
            self.layerOutputs.append(nextInput)
        return nextInput

    def loss(self, prediction, target):
        if target < 0 or target >= prediction.dim[0]:
            raise ValueError("Target index is out of bounds for the prediction matrix.")
        return -1 * math.log(prediction.matrix[target][0])
    
    def backpropogate(self, prediction, target):
        delta_values  = []
        for layer in reversed(range(1, len(self.neuronLayers))):
            if type(layer) == OutputLayer:
                delta_values.append(prediction - Matrix(f"{prediction.dim[0]}x1", *[1 if i == target else 0 for i in range(prediction.dim[0])]))
            elif type(layer) == MiddleLayer:
                # find the delta value for a middle layer that uses sigmoid activation function
                prev_delta = delta_values[-1]
                prev_weights = self.weightLayers[layer + 1].weights
                layer_output = self.layerOutputs[layer - 1]
                delta = (prev_weights.transpose() * prev_delta)*sigmoid_derivative(layer_output)
                delta_values.append(delta)
        delta_values.reverse()


def sigmoid(arr):
    arr = arr.flatten()
    return Matrix(f"{len(arr)}x1", *[1/(1+math.exp(-x)) for x in arr])

def softmax(arr):
    arr = arr.flatten()
    exp = [math.exp(x) for x in arr]
    sum_exp = sum(exp)
    return Matrix(f"{len(arr)}x1", *[x / sum_exp for x in exp])

def sigmoid_derivative(arr):
    arr = arr.flatten()
    return Matrix(f"{len(arr)}x1", *[x * (1 - x) for x in arr])
