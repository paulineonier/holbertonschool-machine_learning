#!/usr/bin/env python3
"""
Defines a Neuron class for binary classification.
"""

import numpy as np


class Neuron:
    """
    Class that defines a single neuron performing binary classification.
    """

    def __init__(self, nx):
        """
        Initialize the neuron.

        Parameters:
        nx (int): Number of input features.

        Raises:
        TypeError: If nx is not an integer.
        ValueError: If nx is less than 1.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be a integer")
        if nx < 1:
            raise ValueError("nx must be positive")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter for weights."""
        return self.__W

    @property
    def b(self):
        """Getter for bias."""
        return self.__b

    @property
    def A(self):
        """Getter for activated output."""
        return self.__A

    def forward_prop(self, X):
        """
        Performs forward propagation.

        Parameters:
        X (np.ndarray): Input data of shape (nx, m).

        Returns:
        np.ndarray: Activated output.
        """
        z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-z))
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost using logistic regression.

        Parameters:
        Y (np.ndarray): True labels of shape (1, m).
        A (np.ndarray): Predicted outputs of shape (1, m).

        Returns:
        float: Cost.
        """
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A) +
                       (1 - Y) * np.log(1.0000001 - A)) / m
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neuron's predictions.

        Parameters:
        X (np.ndarray): Input data of shape (nx, m).
        Y (np.ndarray): True labels of shape (1, m).

        Returns:
        tuple: Prediction array and cost.
        """
        A = self.forward_prop(X)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, self.cost(Y, A)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Performs one pass of gradient descent.

        Parameters:
        X (np.ndarray): Input data of shape (nx, m).
        Y (np.ndarray): True labels.
        A (np.ndarray): Predicted outputs.
        alpha (float): Learning rate.
        """
        m = Y.shape[1]
        dz = A - Y
        dw = np.matmul(dz, X.T) / m
        db = np.sum(dz) / m

        self.__W -= alpha * dw
        self.__b -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the neuron.

        Parameters:
        X (np.ndarray): Input data of shape (nx, m).
        Y (np.ndarray): True labels of shape (1, m).
        iterations (int): Number of training iterations.
        alpha (float): Learning rate.

        Returns:
        tuple: Evaluation after training.
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")

        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for i in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)

        return self.evaluate(X, Y)
