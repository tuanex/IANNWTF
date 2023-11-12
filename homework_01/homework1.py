# TODO
# + CCE_LOSS



from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import random
import numpy as np
import math

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
        for value in range(size):
            input_array.append([data[value + not_value * size][0]])
            label_array.append([data[value][1]])
        return_array.append((input_array, label_array))
    

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
data1 = generator(data, 2)









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


class Sigmoid:
    # No input necessary
    def __init__(self):
        return None

    # Calculate sigmoid with input_matrix of shape (minibatch_size, perceptron_units)
    def call(self, input_matrix):

        # For loop with 'minibatch-size' iterations
        for batch_size in range(input_matrix.shape[0]):

            # For loop with 'input-size' iterations
            for input_size in range(input_matrix.shape[1]):
                input_matrix[batch_size,input_size] = (1 / (1 + np.exp(-input_matrix[batch_size,input_size])))


        return input_matrix



class Softmax:
    # No input necessary
    def __init__(self):
        return None

    # Returns the final quotient of softmax
    # Expect input_matrix of shape [batch_size, 10]
    # Return array of shape [10]
    def call(self, input_matrix):
        """
        to_be = []
        for i in range(len(self.input_array)):
            to_be.append(self.input_array[i] / self.array)
        """
        # For loop with 'minibatch-size' iterations
        for batch_size in range(input_matrix.shape[0]):

            # Sum exponents of array
            array_value = sum(np.exp(input_matrix[batch_size]))

            # for loop with 'input-size' iterations
            for input_size in range(input_matrix.shape[1]):
                input_matrix[batch_size,input_size] = np.exp(input_matrix[batch_size,input_size]) / array_value


        return input_matrix

############################
# Activation functions END #
############################



############
# CCE LOSS #
############

class CCE_Loss:
    def __init__(self):
        return None
    
    # Input of results and tests and calculating CCE Loss 
    # Expect input matrices of size ('minibatch_size', 10)
    # Return vector of loss per sample of size ('minibatch_size', 1)
    def call(self, data_result, target_result):
        result = np.zeros((target_result.shape[0]))

        # For loop with 'minibatch_size' iterations
        for batch_iter in range(target_result.shape[0]):

            # For loop with 'input_size' iterations
            # Here count for target_result
            for target_iter in range(target_result.shape[1]):
                
                # Here count for data_result
                for data_iter in range(target_result.shape[1]):

                    # Sum of 'target' times logarithm of specific element of 'data_result'
                    result[batch_iter] = result[batch_iter] + target_result[batch_iter, target_iter] * np.log(data_result[batch_iter, data_iter])


        # - (1 / (result size) * (sum over result_size (sum over result_size))
        return -(1 / target_result.shape[1]) * result



    # Input of data result and outcome of loss
    # Result of training as data_result of size ('minibatch_size', 1)
    # loss of size ('minibatch_size', 1)
    # Return vector of loss per sample of size ('minibatch_size', 1)
    def backwards(self, data_result, loss):
        
        # Iterate over 'minibatch_size'
        for batch in range(data_result.shape[0]):
            
            # Iterate over each perceptron
            for perceptron in range():
                


        return 


###################
# End of CCE LOSS #
###################








#############
# MLP Layer #
#############

class MLPLayer:

    # input size is number of perceptron units in the previous layer
    # Percepttron units is amount of perceptrons in current layer
    def __init__(self, activation_function, perceptron_units, input_size):

        self.activation_function = activation_function
        self.perceptron_units = perceptron_units
        self.input_size = input_size

        # Initialisation of weights as random numbers with shape (input_size, perceptron_units)
        self.weights = np.random.normal(0, 0.2, (self.input_size, self.perceptron_units))

        # create a vector of biases. Each bias corresponds to one perceptron
        self.bias = np.zeros((self.perceptron_units))

    # n input of shape minibatchsize, input size, and outputs an ndarray of shape minibatchsize, 
    # num units after applying the weight matrix, the bias and the activation function.
    def forward(self, input_matrix):
        # Input matrix of shape (batch_size , input_size)
        # Weights of shape (input_size, perceptron_units)
        # Result_size of shape (batch_size, perceptron_units)
        
        # Dot product of input_matrix and weights
        # Iterating over batches
        output_matrix = np.dot(input_matrix, self.weights)
        output_matrix = output_matrix + self.bias

        output_matrix = self.activation_function.call(output_matrix)
        
        return output_matrix
        

####################
# End of MLP Layer #
####################








###############
# MLP Network #
###############

class Full_MLP:
    MLPs = []
    # Initialize full MLP with number of layers and each layer's number of perceptrons
    # Only hidden layers need to be specified, since input layers and output layers are given to 64 and 10
    def __init__(self, num_layers, array_of_MLPs):
        self.num_layers = num_layers
        
        # Layer 2, using sigmoid, given number of perceptrons and input of size 64
        self.MLPs.append(MLPLayer(sigmoid, array_of_MLPs[0], 64))

        # Layers 3 to second to last, using sigmoid, given number of perceptrons and input of number of perceptrons in previous layer
        for i in range(num_layers):
            self.MLPs.append(MLPLayer(sigmoid, array_of_MLPs[i], array_of_MLPs[i - 1]))

        # Last layer, using softmax with 10 perceptrons and as input number of perceptrons of second to last layer
        self.MLPs.append(MLPLayer(softmax, 10, array_of_MLPs[num_layers - 1]))


######################
# End of MLP Network #
######################






###########
# Testing #
###########
"""
my_norm = np.random.normal(0, 0.2, [2,5])
my_matrix = np.array([[0.01,0.01,0.01,0.96,0.01],[0.01,0.01,0.96,0.01,0.01]])
my_matrix2 = np.array([[0.,0.,0.,1.,0.], [0.,0.,1.,0.,0.]])
soft = softmax()
soft_arr = soft.call(my_norm)
print(soft_arr)

loss = CCE_Loss()
loss_arr = loss.call(my_matrix, my_matrix2)
print(loss_arr)
"""

print(next(data1))

arr = np.random.normal(0., 0.2, [2,3])

sigmoid = Sigmoid()
layer1 = MLPLayer(sigmoid, 2, 3)
print(layer1.forward(arr))


my_matrix = np.array([[1,2],[3,4]])
print(np.dot(my_matrix, my_matrix))