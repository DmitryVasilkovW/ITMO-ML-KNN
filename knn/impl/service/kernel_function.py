import numpy as np


def uniform_function():
    return 1


def gaussian_function(distance):
    return np.exp(-distance ** 2)


def generalized_function(distance):
    return (1 - distance) ** 2


def exponential_function(distance):
    return np.exp(-distance)
