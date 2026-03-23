import numpy as np

# Inputs
X = np.array([1, 2, 3])
weights = np.array([0.2, 0.3, 0.5])
bias = 0.1

# Activation function
def relu(x):
    return max(0, x)

# Perceptron output
output = relu(np.dot(X, weights) + bias)

print("Output:", output)
