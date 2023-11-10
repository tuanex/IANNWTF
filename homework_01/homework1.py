from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import random
import numpy as np
import math

DEBUG = True
def do_anything():
    return None
######################
# Data and Generator #
######################

# Custom functions
# data has the shape: [(image, label), (image, label), (image, label), .... 1797times]
def generator(data, size):
    # shuffle the tuples within the array, this randomizes data
    random.shuffle(data)
    
    return_array = []
    iterating_sub = math.floor(len(data) / size)
    # Split data into 'len(data)/ size' parts, each 'size' long
    for not_value in range(iterating_sub):
        input_array = []
        label_array = []
        print("batch " + not_value)
        for value in range(size):
            input_array.append([size, data[value + not_value * size][0]])
            label_array.append([size,data[value][1]])
        returnArray.append((input_array, label_array))
    

    for ret in return_array:
      yield ret

# Load dataset
digits = load_digits()

# Refactor images from (len, 8,8) to (len, 64)
images_data = digits.images.reshape((len(digits.images), -1))
for i in range(len(images_data)):
    images_data[i] = images_data[i] / np.max(images_data[i])


# Refactor target value from (num) to (0,0,...,0,0)
target_data = np.zeros(shape = (1797, 10))
for i in range(len(images_data)):
    null_vec = np.array([0,0,0,0,0,0,0,0,0,0])
    null_vec[digits.target[i]] = 1
    target_data[i] = null_vec

# Add together to list of tuples
data = []
for i in range(len(images_data)):
    data.append((images_data[i], target_data[i]))

#############################
# End of Data and Generator #
#############################


# Shuffle and split data into batches
# data has the shape: [(image, label), (image, label), (image, label), .... 1797times]
data1 = generator(data, 10)









########################
# Activation functions #
########################
#???????????????????????????????? NECESSARY ????????????????????????????????
"""
class sigmoidFunction:
  def __init__(self):
    pass

# It should expect inputs as ndarrays of
# shape minibatchsize, num units (where num)
  def call(self, input_arrays):
    sigmoid_np()
    pass
"""


class sigmoid:
    # No input necessary
    def __init__(self):
        return None

    # Calculate sigmoid with input_matrix(minibatch_size, perceptron_units)
    def call(self, input_matrix):
        # For loop with 'minibatch-size' iterations
        for i in range(input_matrix.shape[0]):
            # For loop with 'input-size' iterations
            for j in range(input_matrix.shape[1]):
                input_matrix[i,j] = (1 / (1 + np.exp(-input_matrix[i,j])))#
        return input_matrix


class softmax:
    array = 0
    input_array = []

    # Calculate
    def __init__(self, input_array):
        self.input_array = input_array
        to_be = []
        for i in range(len(input_array)):
            to_be.append(np.exp(input_array[i]))
        self.array = sum(to_be)

    # Returns the final quotient of softmax
    # Expect input_matrix of shape [batch_size, 10]
    # Return array of shape [10]
    def call(self, input_matrix):
        to_be = []
        for i in range(len(self.input_array)):
            to_be.append(self.input_array[i] / self.array)
        return to_be

############################
# Activation functions END #
############################









#############
# MLP Layer #
#############

class MLPLayer:

    # input size is number of perceptron units in the previous layer
    # Percepttron units is amount of perceptrons in current layer
    def __init__(self, activation_function, perceptron_units, input_size):

        self.activation_function = activation_function()
        self.perceptron_units = perceptron_units
        self.input_size = input_size

        # Initialisation of weights as random numbers with shape (input_size, perceptron_units)
        self.weights = np.random.normal(0, 0.2, (self.input_size, self.perceptron_units))

        # create a vector of biases. Each bias corresponds to one perceptron
        self.bias = np.zeros((self.perceptron_units))

    # n input of shape minibatchsize, input size, and outputs an ndarray of shape minibatchsize, num units after applying the weight matrix, the bias and the activation function.
    def forward(self, input_matrix):
        # Input matrix of shape (batch_size , input_size)
        # Weights of shape (input_size, perceptron_units)
        # Result_size of shape (batch_size, perceptron_units)
        output_matrix = input_matrix * self.weights
        output_matrix = output_matrix + self.bias

        output_matrix = self.activation_function.call(output_matrix)
        
        return output_matrix
        

####################
# End of MLP Layer #
####################


if DEBUG:
    layer_1 = MLPLayer(sigmoid, 2, 2)
    my_matrix = np.array([[2,2],[3,3]])
    fw = layer_1.forward(my_matrix)
    print(fw)