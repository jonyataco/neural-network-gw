import numpy as np
import random
from neural_network import NeuralNetwork

# Initializing dimensions of the grid
GRID_SIZE = 4
reward_size = -1
terminationState = [GRID_SIZE - 1, GRID_SIZE - 1]
actions = [[0,1], [0, -1], [-1,0], [1,0]]
states = [[i,j] for i in range (GRID_SIZE) for j in range(GRID_SIZE)]

# Returns the next state that is given after taking action
def getStateAndReward(initialState, action):
    # Check if the initial state is equal to the termination state.
    # If so return none.
    if initialState == terminationState:
        return None, 0
    newState = np.array(initialState) + np.array(actions[action])
    # Check if the action puts the agent out of bounds.
    if -1 in newState or GRID_SIZE in newState:
        newState = initialState
    return newState, reward_size

# Setting constants used for training the Neural Network
epochs = 3
gamma = 0.9 # Setting gamma to 0.9 since it may take several moves to goal
epsilon = 1

model = NeuralNetwork()
# Training the network
for i in range(epochs):
    state = [0,0]
    gameInProgress = True
    while (gameInProgress):
        qvals = model.predict(state)
        if (random.random() < epsilon):
            action = np.random.randint(0,4)
        else:
            action = np.argmax(qvals)
        new_state, reward = getStateAndReward(state, action)
        newQVals = model.predict(new_state)
        maxQVal = np.max(newQVals)
        # y represents the target vector
        y = np.zeros(4)
        # Since we only care about updating the weights for the action taken,
        # we are going to create the target that is identical to the output
        # except change the target for the action taken to reward + (gamma*maxQ)

        # Copy the qvals into the target vector
        y[:] = qvals[:]

        # Check if non-terminal state
        if reward == -1:
            update = (reward + (gamma * maxQVal))
        else:
            update = reward

        # Changing the target value for the action taken.
        y[action] = update
        model.train(qvals, target_values)
