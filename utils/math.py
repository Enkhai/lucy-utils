import numpy as np


def cosine_similarity(a, b):
    return ((a / np.linalg.norm(a, axis=-1).reshape((-1, 1))) * (b / np.linalg.norm(b, axis=-1).reshape(-1, 1))).sum(-1)
