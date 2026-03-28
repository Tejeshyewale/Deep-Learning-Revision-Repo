import numpy as np

# Dataset (XOR problem)
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

# Initialize weights
np.random.seed(42)
W1 = np.random.rand(2, 2)
W2 = np.random.rand(2, 1)

# Activation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Training
for epoch in range(5000):
    # Forward
    hidden = sigmoid(np.dot(X, W1))
    output = sigmoid(np.dot(hidden, W2))

    # Error
    error = y - output

    # Backprop
    d_output = error * sigmoid_derivative(output)
    d_hidden = d_output.dot(W2.T) * sigmoid_derivative(hidden)

    # Update
    W2 += hidden.T.dot(d_output)
    W1 += X.T.dot(d_hidden)

print("Output:\n", output)
