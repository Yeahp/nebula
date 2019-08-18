import numpy as np


def cost_derivative(output_activations, y):
    return output_activations - y


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1.0 - sigmoid(z))


if __name__ == "__main__":
    input_1 = np.random.randn(2, 1)
    print(input_1)

    input_2 = np.random.randn(2, 1)
    print(input_2)

    #print(sigmoid(input))

    #print(sigmoid_prime(input))

    print(cost_derivative(input_1, input_2))

    print(sigmoid_prime(input_1))

    print(cost_derivative(input_1, input_2) * sigmoid_prime(input_1))