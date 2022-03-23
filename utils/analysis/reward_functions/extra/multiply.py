import numpy as np


def multiply(rewards):
    return np.array(rewards).prod(0)
