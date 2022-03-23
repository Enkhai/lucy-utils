from typing import List

import numpy as np


def anneal(rewards: List[np.ndarray], x_fading_steps: List[int]):
    assert len(rewards) == len(x_fading_steps) + 1
    assert sum(len(r) for r in rewards) == sum(x_fading_steps) * 2

    kernels = [np.linspace(0, 1, step) for step in x_fading_steps]
    inv_kernels = [k[::-1] for k in kernels]

    # First reward array * inverse kernel + second reward array * kernel
    # The weights of the kernel and the inverse kernel add up to 1
    reward_array = [rewards[i][-len(kernels[i]):] * inv_kernels[i] +
                    rewards[i + 1][:len(kernels[i])] * kernels[i]
                    for i in range(len(rewards) - 1)]

    return np.concatenate(reward_array)
