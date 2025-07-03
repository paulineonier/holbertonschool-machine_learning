#!/usr/bin/env python3
"""
DeepNeuralNetwork class with save/load methods.
"""

import numpy as np
import pickle
import os


class DeepNeuralNetwork:
    # (ton __init__ et autres méthodes ici)

    def save(self, filename):
        """
        Saves the instance object to a pickle file.
        Adds '.pkl' extension if not present.

        Args:
            filename (str): File name to save the object.
        """
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        Loads a pickled DeepNeuralNetwork object from a file.

        Args:
            filename (str): File name to load the object from.

        Returns:
            DeepNeuralNetwork instance or None if file does not exist.
        """
        if not os.path.exists(filename):
            return None
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
        return obj
