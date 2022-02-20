import numpy as np


def sequential(rewards):
    return np.concatenate(rewards, 0)
