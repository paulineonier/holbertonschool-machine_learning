#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


class DeepNeuralNetwork:
    def __init__(self, nx, layers):
        # Vérifications des arguments
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0 or not all(
            isinstance(x, int) and x > 0 for x in layers
        ):
            raise TypeError("layers must be a list of positive integers")

        self.nx = nx
        self.layers = layers
        self.L = len(layers)  # nombre de couches
        self.cache = {}
        self.weights = {}

        # Initialisation des poids avec He et biais à 0
        for l in range(self.L):
            layer_size = layers[l]
            prev_layer_size = nx if l == 0 else layers[l - 1]
            self.weights["W" + str(l + 1)] = (
                np.random.randn(layer_size, prev_layer_size)
                * np.sqrt(2 / prev_layer_size)
            )
            self.weights["b" + str(l + 1)] = np.zeros((layer_size, 1))

    def forward_prop(self, X):
        self.cache["A0"] = X
        for l in range(self.L):
            Wl = self.weights["W" + str(l + 1)]
            bl = self.weights["b" + str(l + 1)]
            Al_prev = self.cache["A" + str(l)]
            Zl = np.matmul(Wl, Al_prev) + bl
            if l == self.L - 1:
                # Sigmoid activation à la sortie
                Al = 1 / (1 + np.exp(-Zl))
            else:
                # ReLU pour couches cachées
                Al = np.maximum(0, Zl)
            self.cache["A" + str(l + 1)] = Al
        return Al, self.cache

    def cost(self, Y, A):
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        )
        return cost

    def evaluate(self, X, Y):
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, Y, alpha=0.05):
        m = Y.shape[1]
        weights_copy = self.weights.copy()
        L = self.L
        dZ = 0

        for l in reversed(range(1, L + 1)):
            A_curr = self.cache["A" + str(l)]
            A_prev = self.cache["A" + str(l - 1)]

            if l == L:
                dZ = A_curr - Y
            else:
                W_next = weights_copy["W" + str(l + 1)]
                dZ_next = dZ
                dZ = np.matmul(W_next.T, dZ_next)
                dZ = dZ * np.where(A_curr > 0, 1, 0)  # dérivée ReLU

            dW = (1 / m) * np.matmul(dZ, A_prev.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

            self.weights["W" + str(l)] = self.weights["W" + str(l)] - alpha * dW
            self.weights["b" + str(l)] = self.weights["b" + str(l)] - alpha * db

    def train(
        self, X, Y, iterations=5000, alpha=0.05,
        verbose=True, graph=True, step=100
    ):
        # Validations des arguments
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")

        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        if (verbose or graph):
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []

        for i in range(iterations + 1):
            A, _ = self.forward_prop(X)
            cost = self.cost(Y, A)

            if verbose and (i % step == 0 or i == 0 or i == iterations):
                print(f"Cost after {i} iterations: {cost}")

            if graph and (i % step == 0 or i == 0 or i == iterations):
                costs.append((i, cost))

            if i < iterations:
                self.gradient_descent(Y, alpha)

        if graph:
            x_vals, y_vals = zip(*costs)
            plt.plot(x_vals, y_vals, 'b-')
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)
