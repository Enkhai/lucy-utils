def diff(reward, negative_slope=1):
    """
    Difference reward function. Not to be confused with potential-based reward shaping function.
    First values are past rewards, last values are recent rewards
    """
    rew = reward[1:] - reward[:-1]
    rew[rew < 0] *= negative_slope
    return rew
