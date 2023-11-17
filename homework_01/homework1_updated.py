from sklearn.datasets import load_digits
import numpy as np


digits = load_digits(return_X_y=True)





######################
# Generator Function #
######################

def generator(size):
    
    # Split digits into data and label
    data, label = digits

    # Arrange array into one dimensional floats
    data.reshape(-1)

    for i in range(len(data)):
        data[i] = data[i] / data[i].max()

    # shuffle the tuples within the array, this randomizes data
    perm = np.random.permutation(len(data))
    
    data = data[perm]
    label = label[perm]

    l_onehot = np.zeros((1797, 10))
    # One hottify label
    for l in range(label.shape[0]):
        l_onehot[l,label[l]] = 1
    
    label = l_onehot

    # Split data and labels into batches
    split_into = np.floor(data.shape[0] / size).astype(np.int32)

    split_data = np.array_split(data, split_into)
    split_label = np.array_split(label, split_into)

    for dat, lab in zip(split_data, split_label):
        yield dat, lab





########################
# Activation Functions #
########################

class Sigmoid:

    def __init__(self):
        pass


    def call(self, input_matrix):
        output_matrix = 1 / (1 + np.exp(-input_matrix))
        return output_matrix


    def backwards(self, input_matrix):

        # Derivative of the Sigmoid activation function
        dactivation_dpreactivation = self.call(input_matrix) * (1 - self.call(input_matrix))

        return dactivation_dpreactivation



class Softmax:

    def __init__(self):
        pass

    def call(self, input_matrix):
        # For loop with 'minibatch-size' iterations
        for batch_size in range(input_matrix.shape[0]):

            # Sum exponents of array
            array_value = sum(np.exp(input_matrix[batch_size]))

            # for loop with 'input-size' iterations
            for input_size in range(input_matrix.shape[1]):
                input_matrix[batch_size,input_size] = np.exp(input_matrix[batch_size,input_size]) / array_value

        return input_matrix





##########
# CCE Loss
##########

class CCE_Loss:

    def __init__(self):
        pass

    
    # Take results of MLP and the actual label and calculate losses
    def call(self, result_label, actual_label):

        result = actual_label * np.log(result_label)

        new_result = np.zeros(result.shape[0])
        for batch in range(len(result)):

            new_result[batch] = -1 * np.sum(result[batch])

        return new_result

    
    # Take results of MLP and the actual label and calculate derivative of losses
    def backwards(self, result_label, actual_label):

        # Derivative of Lcce Loss
        result = result_label - actual_label
        
        return result
        
        



#######
# MLP #
#######

class MLP:
    
    def __init__(self, activation_function, output_size, input_size, last_layer):
        
        self.activation_function = activation_function
        self.output_size = output_size
        self.input_size = input_size

        # Initialize weights as random values from a normal distribution
        self.weights = np.random.normal(0, 0.2, (self.input_size, self.output_size))
        self.bias = np.zeros((output_size))

        self.last_layer = last_layer

        
    # Take an input of shape (input_size, output_size)
    def forward(self, input_matrix):

        self.input_matrix = input_matrix

        # Calculate preactivation
        output_matrix = np.matmul(input_matrix, self.weights)
        output_matrix = output_matrix + self.bias

        # Calculate activation with preactivation
        self.output_matrix = self.activation_function.call(output_matrix)

        return output_matrix


    def backwards(self, feedback, target):
        
        # Get dL / dPreactivation
        # Check for last layer
        if self.last_layer == True:
            # Apply loss function
            dL_dP = CCE_Loss().backwards(self.output_matrix, target)

            # Calculate feedback to next layer
            # Get dL/dI = dL/dP * dP/dI (i.e. weights)
            dL_dI = np.matmul(dL_dP, np.transpose(self.weights))
            
            # Calculate change in weights
            # Get dL/dW = dL/dP * dP/dW (i.e. input)
            dL_dW = np.transpose(self.input_matrix) @ dL_dP

        else:
            # Apply backwards sigmoid function
            dL_dP = self.activation_function.backwards(feedback)

            # Calculate feedback to next layer
            # Get dL/dI = dL/dP * dP/dI (i.e. weights)
            dL_dI = np.matmul(dL_dP, np.transpose(self.weights))

            # Calculate change in weights
            # Get dL/dW = dL/dP * dP/dW (i.e. input)
            dL_dW = np.transpose(self.input_matrix) @ dL_dP
        
        
        # Apply changes to weights and bias
        self.weights = self.weights - dL_dW
        self.bias = self.bias - np.mean(dL_dP, axis = 0)

        # Return difference to previous layer
        return dL_dI



class Full_MLP:

    def __init__(self, hidden_layers):

        self.MLP = []
        self.num_layers = len(hidden_layers) + 1


        # First hidden layer
        self.MLP.append(MLP(Sigmoid(), hidden_layers[0], 64, False))


        # Second to last hidden layer
        for layer in range(1, len(hidden_layers)):
            
            this_layer = hidden_layers[layer]
            prev_layer = hidden_layers[layer - 1]

            self.MLP.append(MLP(Sigmoid(), this_layer, prev_layer, False))

        
        # Output layer
        self.MLP.append(MLP(Softmax(), 10, hidden_layers[-1], True))

    # Combine forward step for each layer of full MLP
    def forward(self, input_matrix):
        
        new_input_matrix = self.MLP[0].forward(input_matrix)

        for layer in range(1, self.num_layers):
            new_input_matrix = self.MLP[layer].forward(new_input_matrix)

        return new_input_matrix

    # Combine backward step for each layer of full MLP
    def backwards(self, feedback, target):
        
        new_feedback = self.MLP[self.num_layers - 1].backwards(feedback, target)

        for layer in range(self.num_layers -2, -1, -1):
            
            new_feedback = self.MLP[layer].backwards(new_feedback, target)
        
        return new_feedback





############
# TRAINING #
############

def training(FMLP, epochs, batch_size):

    # Iterate over epochs
    for epoch in range(epochs):

        for dat, lab in generator(batch_size):

            # Forward path
            end_of_for = full_mlp.forward(dat)

            # Backward path
            end_of_back = full_mlp.backwards(end_of_for, lab)

    return full_mlp



# Give hidden layers
full_mlp = Full_MLP([64, 64, 32, 32])

# Train Multi-layer perceptron
trained_mlp = training(full_mlp, 2, 8)
