import numpy as np
import random
from keras.models import Sequential
from keras.layers import Layer, Dense
from keras.optimizers import RMSprop

# Initializing dimensions of the grid
GRID_SIZE = 4
reward_size = -1
terminationState = np.array([GRID_SIZE - 1, GRID_SIZE - 1])
actions = [[0,1], [0, -1], [-1,0], [1,0]]

def printGrid(state):
    grid = np.zeros((4,4), dtype=str)
    grid[3,3] = 'G'
    grid[state[0], state[1]] = 'A'
    print(grid)

# Returns the next state that is given after taking action
def getStateAndReward(initialState, action):
    # Check if the initial state is equal to the termination state.
    # If so return none.
    if np.array_equal(initialState, terminationState):
        return np.array([3,3]), 0
    newState = np.array(initialState) + np.array(actions[action])
    # Check if the action puts the agent out of bounds.
    if -1 in newState or GRID_SIZE in newState:
        newState = initialState
    return newState, reward_size

def testNetwork(network):
    i = 0
    state = np.array([0,0])
    print("Initial State")
    printGrid(state)
    gameInProgress = True
    while (gameInProgress):
        qvals = model.predict(state.reshape(1,2), batch_size=1)
        action = np.argmax(qvals)
        #print(f'Move {i}. Taking action {action}')
        print(f'Move {i}.')
        print(f'Q values at current state: \n Up: {qvals[0][0]}, Down: {qvals[0][1]}, Left: {qvals[0][2]}, Right: {qvals[0][3]}')
        state, reward = getStateAndReward(state, action)
        printGrid(state)
        if reward != -1:
            gameInProgress = False
            print('Done!')
        i += 1
        if (i > 10):
            print("Not trained enough")
            break

# Setting constants used for training the Neural Network
epochs = 1000
gamma = 0.9 # Setting gamma to 0.9 since it may take several moves to goal
epsilon = 1

# Configuring the network with how many hidden layers and the shape 
# of the output layer
model = Sequential()
model.add(Dense(units=8, activation='sigmoid'))
model.add(Dense(units=8, activation='sigmoid'))
model.add(Dense(units=4, activation='linear'))
model.compile(loss='mse', optimizer='rmsprop')

# Training the network
for i in range(epochs):
    print(f'Game{i}')
    state = np.array([0, 0])
    gameInProgress = True
    while (gameInProgress):
        qvals = model.predict(state.reshape(1,2), batch_size=1)
        if (random.random() < epsilon):
            action = np.random.randint(0,4)
        else:
            action = np.argmax(qvals)
        new_state, reward = getStateAndReward(state, action)
        newQVals = model.predict(new_state.reshape(1,2), batch_size=1)
        maxQVal = np.max(newQVals)
        # y represents the target vector
        y = np.zeros(4)
        # Since we only care about updating the weights for the action taken,
        # we are going to create the target that is identical to the output
        # except change the target for the action taken to reward + (gamma*maxQ)

        # Copy the qvals into the target vector
        y = np.zeros((1,4))
        y[:] = qvals[:]

        # Check if non-terminal state
        if reward == -1:
            update = (reward + (gamma * maxQVal))
        else:
            update = reward

        # Changing the target value for the action taken.
        y[0][action] = update
        model.fit(state.reshape(1,2), y, batch_size=1, verbose=0)
        state = new_state

        if reward != -1:
            gameInProgress = False

    if epsilon > 0.1:
        epsilon -= (1/epochs)

testNetwork(model)
