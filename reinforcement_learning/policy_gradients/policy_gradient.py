#!/usr/bin/env python3
import numpy as np


def policy(matrix, weight):
    """Computes the policy using a weight matrix."""
    z = np.matmul(matrix, weight)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=1, keepdims=True)
