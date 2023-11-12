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

        # Iterating over number of samples
        for batch_size in range(input_matrix.shape[0]):

            # Iterating over number of inputs
            for input_size in range(input_matrix.shape[1]):
                input_matrix[batch_size,input_size] = (1 / (1 + np.exp(-input_matrix[batch_size,input_size])))

        return input_matrix

    # Preactivation of size ('minibatch_size', num_units)
    # Activation of size ('minibatch_size', num_units)
    # Error signal (dL / d activation) of size ('num_units', 1)
    def backwards(self, preactivation, activation, error_signal):
        dL_dpreactivation = np.zeros((preactivation.shape[0],preactivation.shape[1]))

        # Iterating over number of samples
        for batch_iter in range(preactivation.shape[0]):

            # Iterating over number of inputs
            for input_iter in range(preactivation.shape[1]):
                dL_dpreactivation[batch_iter, input_iter] = np.exp(-preactivation[batch_iter, input_iter]) / (1 + np.exp(-preactivation[batch_iter, input_iter]))


        return dL_dpreactivation
    



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
        result = np.zeros(data_result.shape[0])

        # For loop with 'minibatch_size' iterations
        for batch_iter in range(target_result.shape[0]):

            # For loop with 'input_size' iterations
            # Here count for target_result
            for result_iter in range(target_result.shape[1]):
                # Sum of 'target' times logarithm of specific element of 'data_result'
                result[batch_iter] = target_result[batch_iter, result_iter] * np.log(data_result[batch_iter, result_iter])


        # (sum over result_size (sum over result_size))
        return -1 * result



    # Input of data result and outcome of loss
    # Result of training as data_result of size ('minibatch_size', 1)
    # loss of size ('minibatch_size', 1)
    # Return vector of loss per sample of size ('minibatch_size', 1)
    def backwards(self, data_result, target_result, loss):
        output_matrix = np.zeros((data_result.shape[0], data_result.shape[1]))

        # Iterate over 'minibatch_size'
        for batch in range(data_result.shape[0]):
            
            # Iterate over each perceptron
            for perceptron in range(data_result.shape[1]):
                output_matrix[batch, perceptron] = data_result[batch, perceptron] - target_result[batch, perceptron]


        return output_matrix


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
        print("input", input_matrix)
        output_matrix = np.dot(input_matrix, self.weights)
        output_matrix = output_matrix + self.bias

        output_matrix = self.activation_function.call(output_matrix)
        
        return output_matrix
        
    
    # Expect dL_dpreactivation of size ('minibatch_size', 'num_units')
    # Expect preactivation presenting both weight and input 
    # Weight of size ('input_size', 'perceptron_units')
    # Input of size ('input_size', 'perceptron_size')
    def weights_backward(self, dL_dpreactivation, weight, prev_lay):
        dL_dW_dinput = np.array([0.,0.])

        # Calculate dL/dW
        # Matrix of shape ('minibatch_size', 'perceptron_units)
        dL_dW_dinput[0] = dL_dpreactivation * self.weight

        # Calculate dL/dInput
        # Matrix of shape ('minibatch_size', 'input_size')
        dL_dW_dinput[1] = dL_dpreactivation * self.prev_lay

        # Return [dW | dinput]
        return dL_dW_dinput

    
    def layer_backwards(self, preactivation, activation, error_signal):

        activation_back = self.activation_function.backwards(preactivation, activation, error_signal)
        weights = self.weights_backward(activation_back, self.weights, self.input_matrix)
        return activation_back, weights
        

####################
# End of MLP Layer #
####################








###############
# MLP Network #
###############

class Full_MLP:
    # List of MLPs
    MLPs = []
    # Dictionary with everything pertaining to preactivations and activations
    # With a shape of ('layer', ['preactivation','activation'])
    input_dict = {}
    # Initialize full MLP with number of layers and each layer's number of perceptrons
    # Only hidden layers need to be specified, since input layers and output layers are given to 64 and 10
    def __init__(self, num_layers, array_of_MLPs):
        self.num_layers = num_layers
        
        # Layer 2, using sigmoid, given number of perceptrons and input of size 64
        self.MLPs.append(MLPLayer(Sigmoid(), array_of_MLPs[0], 64))
        self.input_dict.update({"Inputs Layer 0": 0})

        # Layers 3 to second to last, using sigmoid, given number of perceptrons and input of number of perceptrons in previous layer
        for i in range(num_layers):
            self.MLPs.append(MLPLayer(Sigmoid(), array_of_MLPs[i], array_of_MLPs[i - 1]))
            self.input_dict.update({"Inputs Layer "+ str(i): 0})

        # Last layer, using softmax with 10 perceptrons and as input number of perceptrons of second to last layer
        self.MLPs.append(MLPLayer(Softmax(), 10, array_of_MLPs[num_layers - 1]))        
        self.input_dict.update({"Inputs Layer " + str(num_layers - 1): 0})

    def full_MLP_forward(self):

        for mlp in range(self.num_layers + 1):
            self.MLPs[mlp].forward(self.input_dict["Inputs Layer " + str(mlp)])
    
    def full_MLP_backward(self):
        
        for mlp in range(self.num_layers - 1, -1, -1):

            self.MLPs[mlp].layer_backwards(self.input_dict["Input Layer " + str(mlp)][0], self.input_dict["Input Layer " + str(mlp)][1], self.input_dict["Input Layer " + str(mlp + 1)])
            # Idea: Receive values for backpropagation and store for fitting in training()
            MLP.input_dict["Inputs Layer ", str(mlp)].append(self.MLPs[mlp].layer_backwards(self.input_dict["Input Layer " + str(mlp)][0], self.input_dict["Input Layer " + str(mlp)][1], self.input_dict["Input Layer " + str(mlp + 1)]))
        

######################
# End of MLP Network #
######################








#####################
# Training Function #
#####################

def training(epochs, MLP, batches):
    for i in range(epochs):
        for batch in batches:
            # Put in start values
            MLP.input_dict.update({"Inputs Layer 1": batch})

            # Calculate values through to the end until output layer
            MLP.full_MLP_forward()

            # Calculate Loss
            loss = CCE_Loss()
            loss.call()

            # Backpropagation through till beginning
            MLP.full_MLP_backward()

            # Fitting
            """
            for layer in MLP_layers:
                layers.weights = layers.weights + MLP.input_dict
            """


    return MLP
############################
# End of Training Function #
############################











###########
# Testing #
###########

size = 100
zip_gen = generator(data, 2)

print(zip_gen)
full_mlp = Full_MLP(2, [10, 10])

data_in_batches, label_in_batches = zip(*zip_gen)

data_in_batches = np.array(data_in_batches)
label_in_batches = np.array(label_in_batches)

print(data_in_batches)

print(training(1, full_mlp, data_in_batches))







"""# Sigmoid instance
sigmoid = Sigmoid()

# Softmax instane
soft = Softmax()

# CCE_Loss instance
loss = CCE_Loss()

# MLP instance
mlp = MLPLayer(soft, 2, 5)

my_norm = np.random.normal(0, 0.2, [2,5])
my_matrix = np.array([[0.01,0.01,0.01,0.96,0.01],[0.01,0.01,0.96,0.01,0.01]])
my_matrix2 = np.array([[0.,0.,0.,1.,0.], [0.,0.,1.,0.,0.]])

# Loss call -> distributes to 1
soft_arr = soft.call(my_norm)

# Loss call -> Loss of softmax.vec
loss_call = loss.call(my_matrix, my_matrix2)
#print(loss_call)

# Loss backwards -> Loss-func backwards to
loss_back = loss.backwards(my_matrix, my_matrix2, loss_call)
#print(loss_back)

# Sigmoid call
sig_arr = sigmoid.call(my_norm)
print(sig_arr)
print(sigmoid.backwards(my_norm, sig_arr, loss_back))





full = Full_MLP(3, [32, 20, 10])
#full.full_MLP_backward()

dict = {
    "hello": [1,2]
}

print(dict["hello"][0])
print(dict["hello"][1])"""