import numpy as np


def cosine_similarity(a, b):
    return ((a / (np.linalg.norm(a, axis=-1).reshape(-1, 1) + 1e-8)) *
            (b / (np.linalg.norm(b, axis=-1).reshape(-1, 1) + 1e-8))).sum(-1)
