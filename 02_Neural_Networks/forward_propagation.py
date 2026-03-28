import numpy as np

# Input
X = np.array([[1, 2]])

# Weights
W1 = np.array([[0.5, 0.2],
               [0.3, 0.7]])

# Bias
b1 = np.array([0.1, 0.2])

# Activation
def relu(x):
    return np.maximum(0, x)

# Forward propagation
Z1 = np.dot(X, W1) + b1
A1 = relu(Z1)

print("Output:", A1)
