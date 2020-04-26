import numpy as np
import random

LEARNING_RATE = .2

# Performs sigmoid operation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Scales the inputs
def scale_inputs(input_vector):
    for i in range(len(input_vector)):
        input_vector[i] = (input_vector[i] / 4) + .25
    return input_vector

''' Class that creates a neural network \
    The Neural Network will have two hidden layers with 8 nodes each '''
class NeuralNetwork():
    ''' There will be three sets of weights for the network.
        The first set of weights will be a 8 x 3 matrix.
        The second set of weights will be a 8 x 9 matrix.
        The last set of weights will be a 9 x 4 matrix.
        The np.random.rand function returns an array of a given dimension
        between [0,1) '''
    def __init__(self):
        # Appending a bias term with value of  1 to the input vector.
        self.weights_1 = np.random.rand(8,3)
        self.weights_2 = np.random.rand(8,9)
        self.weights_3 = np.random.rand(4,9)

    ''' Given an input_vector, returns output nodes '''
    def predict(self, input_vector):
        # Appending bias node to input vector
        input_vector = scale_inputs(input_vector)
        print(input_vector)
        input_vector = np.append(input_vector, 1)
        print(input_vector)

        # FIRST HIDDEN LAYER
        first_layer = np.zeros(8)
        print(f'First_layer before going through activation function:\n'
              f'{first_layer}')
        # Performing calculations to get values of neurons in first layer
        first_layer = sigmoid(np.dot(self.weights_1, input_vector))
        print(f'First_layer after going through activation function:\n'
              f'{first_layer}')

        # SECOND HIDDEN LAYER
        # Appending bias node
        first_layer = np.append(first_layer, 1)
        # Performing calculations to get values of neurons in second layer
        second_layer = np.zeros(8)
        print(f'Second_layer before going through activation function:\n'
              f'{second_layer}')
        second_layer = sigmoid(np.dot(self.weights_2, first_layer))
        print(f'Second after going through activation function:\n'
              f'{second_layer}')

        # OUTPUT LAYER
        # Appending bias node
        second_layer = np.append(second_layer, 1)
        # Performing calculations to get values of neurons in output layer
        output_layer = np.zeros(4)
        print(self.weights_3)
        print(f'output_layer before going through activation function:\n'
              f'{output_layer}')
        output_layer = sigmoid(np.dot(self.weights_3, second_layer))
        print(f'output_layer after going through activation function:\n'
              f'{output_layer}')

        return output_layer

    ''' Trains the neural network given the target values '''
    # NOT FINISHED
    def train(self, output_vector, target_values):
        errors = np.zeros(4)
        for i in range(len(output_vector)):
            errors[i] = output_vector[i](1 - output_vector[i])(target_values[i] - output_vector[i])
            delta = LEARNING_RATE * errors[i] * (output_vector[i])
            self.weights_2 = self.weights_2

model = NeuralNetwork()
output = model.predict([1,1])
