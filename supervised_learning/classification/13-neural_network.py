#!/usr/bin/env python3
"""Neural Network with one hidden layer performing binary classification"""
import numpy as np


class NeuralNetwork:
    """Defines a neural network with one hidden layer"""

    def __init__(self, nx, nodes):
        """
        Class constructor
        nx: number of input features
        nodes: number of nodes in hidden layer
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # Hidden layer
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0

        # Output layer
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    # Getters
    @property
    def W1(self):
        return self.__W1

    @property
    def b1(self):
        return self.__b1

    @property
    def A1(self):
        return self.__A1

    @property
    def W2(self):
        return self.__W2

    @property
    def b2(self):
        return self.__b2

    @property
    def A2(self):
        return self.__A2

    def forward_prop(self, X):
        """
        Calculates forward propagation of the neural network
        X: numpy.ndarray of shape (nx, m) with input data
        Returns: A1, A2
        """
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))

        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))

        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        Calculates cost using logistic regression
        Y: correct labels, shape (1, m)
        A: activated output, shape (1, m)
        Returns: cost
        """
        m = Y.shape[1]
        cost = - (1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        )
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates neural network’s predictions
        X: input data, shape (nx, m)
        Y: correct labels, shape (1, m)
        Returns: prediction (1 or 0), cost
        """
        _, A2 = self.forward_prop(X)
        prediction = np.where(A2 >= 0.5, 1, 0)
        cost = self.cost(Y, A2)
        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network
        Updates private attributes __W1, __b1, __W2, and __b2

        X: input data, shape (nx, m)
        Y: correct labels, shape (1, m)
        A1: activated output of hidden layer, shape (nodes, m)
        A2: predicted output, shape (1, m)
        alpha: learning rate
        """
        m = Y.shape[1]

        dZ2 = A2 - Y  # (1, m)
        dW2 = (1 / m) * np.matmul(dZ2, A1.T)  # (1, nodes)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)  # (1,1)

        g1 = A1 * (1 - A1)  # sigmoid derivative
        dZ1 = np.matmul(self.__W2.T, dZ2) * g1  # (nodes, m)
        dW1 = (1 / m) * np.matmul(dZ1, X.T)  # (nodes, nx)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)  # (nodes,1)

        self.__W1 -= alpha * dW1
        self.__b1 -= alpha * db1
        self.__W2 -= alpha * dW2
        self.__b2 -= alpha * db2

    def train(self, X, Y, iterations=1, alpha=0.05):
        """
        Trains neural network, no loop: one forward prop + one gradient descent
        X: numpy.ndarray with shape (nx, m)
        Y: numpy.ndarray with shape (1, m)
        iterations: must be 1 here, no loops allowed
        alpha: learning rate
        Returns: evaluation of training data after one update
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations != 1:
            raise ValueError("iterations must be 1 (no loop version)")

        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        A1, A2 = self.forward_prop(X)
        self.gradient_descent(X, Y, A1, A2, alpha)

        return self.evaluate(X, Y)
