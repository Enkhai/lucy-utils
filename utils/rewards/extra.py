from rlgym.utils import RewardFunction
from rlgym_tools.extra_rewards import diff_reward


class DiffPotentialReward(diff_reward.DiffReward):
    """
    Potential-based reward shaping function with a `negative_slope` magnitude parameter
    """

    def __init__(self, reward_function: RewardFunction, gamma=0.99, negative_slope=1.):
        super(DiffPotentialReward, self).__init__(reward_function, negative_slope)
        self.gamma = gamma

    def _calculate_diff(self, player, rew):
        last = self.last_values.get(player.car_id)
        self.last_values[player.car_id] = rew
        if last is not None:
            ret = (self.gamma * rew) - last
            return self.negative_slope * ret if ret < 0 else ret
        else:
            return 0
