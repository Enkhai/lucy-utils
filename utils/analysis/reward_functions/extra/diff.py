def diff(reward, gamma=1):
    """
    Potential reward function (not potential-based reward shaping function)\n
    First values are past rewards, last values are recent rewards
    """
    return reward[1:] - gamma * reward[:-1]
