import numpy as np


def distribute(rewards, team_idcs, team_spirit=0.3):
    reward_lengths = np.array([len(r) for r in rewards])
    # Assert all rewards have the same length
    assert (reward_lengths == (sum(reward_lengths) / len(reward_lengths))).sum() == len(reward_lengths)

    team_idcs = np.array(team_idcs, dtype=bool)
    rewards = np.stack(rewards)
    blue_means = rewards[~team_idcs].mean(0)
    orange_means = rewards[team_idcs].mean(0)

    blue = blue_means * team_spirit + rewards[~team_idcs] * (1 - team_spirit) - orange_means
    orange = orange_means * team_spirit + rewards[team_idcs] * (1 - team_spirit) - blue_means

    final_rewards = np.zeros_like(rewards)
    final_rewards[~team_idcs] = blue
    final_rewards[team_idcs] = orange

    return final_rewards
