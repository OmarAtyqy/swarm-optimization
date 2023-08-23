import numpy as np
import warnings
import pandas as pd
from TargetFunction import TargetFunction
from WhaleOptimizer import WhaleOptimizer
from time import time


# create a simple neural network with 4 inputs, 2 hidden layers with 10 neurons each and 1 output layer with 3 neurons
# the goal is to classify data from the iris dataset

# choose the number of epochs to train for
n_iter = 100

# define the NN architecture
n_input = 4
n_hidden1 = 10
n_hidden2 = 10
n_output = 3

# define the optimizer parameters
N_pods = 20
N_whales = 100

# ======================================================================================================================

# define the total number of weights
n_weights = (n_input * n_hidden1) + (n_hidden1 * n_hidden2) + (n_hidden2 * n_output) + n_hidden1 + n_hidden2 + n_output

# initialize the weights and biases
# we're going to use a 1D array for the WHO optimizer
W = np.random.randn(n_weights)

# define the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# define the softmax function
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# define the forward pass to calculate the output
def forward_pass(W, x):
    
    # get the weights for each layer
    W1 = W[0:((n_input+1) * n_hidden1)].reshape(n_input+1, n_hidden1)
    W2 = W[((n_input+1) * n_hidden1):((n_input+1 )* n_hidden1) + ((n_hidden1+1) * n_hidden2)].reshape(n_hidden1+1, n_hidden2)
    W3 = W[((n_input+1) * n_hidden1) + ((n_hidden1+1) * n_hidden2):].reshape(n_hidden2+1, n_output)

    # calculate the output of each layer
    z1 = np.dot(x, W1)
    a1 = sigmoid(z1)
    
    # add a column of ones to the output of the first layer
    a1 = np.insert(a1, 0, 1, axis=1)

    z2 = np.dot(a1, W2)
    a2 = sigmoid(z2)

    # add a column of ones to the output of the second layer
    a2 = np.insert(a2, 0, 1, axis=1)

    z3 = np.dot(a2, W3)
    a3 = softmax(z3)

    return a3

# define the loss function
def loss(y_pred, y_true):
    # Avoid divide by zero error
    epsilon = 1e-7
    
    # Calculate categorical cross-entropy loss
    loss = -np.sum(y_true * np.log(y_pred + epsilon))
    
    return loss

# ======================================================================================================================

# load the iris dataset
iris = pd.read_csv('iris.csv')

# remove the id column
iris = iris.drop('Id', axis=1)

# convert the species column to a numerical value using one hot encoding
iris = pd.get_dummies(iris, columns=['Species'])

# convert the values to floats
iris = iris.astype('float32')

# shuffle the dataset
iris = iris.sample(frac=1).reset_index(drop=True)

# add a column of ones to the dataset to account for the bias
iris.insert(0, 'Bias', 1)

# convert the dataframe to a numpy array
iris = iris.to_numpy()

# split the dataset into training and testing sets
X_train = iris[:100, :-3]
y_train = iris[:100, -3:]
X_test = iris[100:, :-3]
y_test = iris[100:, -3:]

# ======================================================================================================================

# ignore warnings
warnings.filterwarnings('ignore')

# create the target function
def target_function(w):
    # calculate the output of the network
    y_pred = forward_pass(w, X_train)

    # calculate the loss
    loss_value = loss(y_pred, y_train)

    return loss_value

# create the target function object
target = TargetFunction(function=target_function,
                        lowerBound=np.array([-100 for _ in range(n_weights)]),
                        upperBound=np.array([100 for _ in range(n_weights)]),
                        d=n_weights,
                        maxIter=n_iter)

# create the optimizer
WHO = WhaleOptimizer(target,
                  N_pods=N_pods,
                  N_whales=N_whales)

# run the optimizer
start = time()
WHO.find_optimum_threaded(verbose=True)
# WHO.find_optimum(verbose=True)
end = time()

# get the results
W_best, L_best = WHO.get_best(verbose=True)

# ======================================================================================================================

# calculate the output of the network
y_pred = forward_pass(W_best, X_test)

# print the accuracy as percentage with 2 decimal places
print('Accuracy: {:.2f}%'.format(np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1)) / len(y_test) * 100))
print(f"N of pods: {N_pods} | N of whales: {N_whales} | N of iterations per pod: {n_iter}")
print(f"Time: {end - start}s")

# save the weights
if (input("Save weights? (y/n): ") == "y"):
    np.save(arr=W_best, file="weights.npy")