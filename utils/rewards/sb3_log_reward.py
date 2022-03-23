import numpy as np
from rlgym.utils import RewardFunction, common_values
from rlgym.utils.gamestates import PlayerData, GameState
from stable_baselines3.common.logger import Logger


class SB3NamedBlueLogReward(RewardFunction):
    """
    Simple reward function for logging blue individual rewards to a custom logger. Logging for the blue team only
    is important, since blue and orange rewards can often add up to 0.
    """

    def __init__(self, logger: Logger,
                 reward_function: RewardFunction,
                 reward_name: str,
                 ):
        super(SB3NamedBlueLogReward, self).__init__()
        self.logger = logger
        self.reward_function = reward_function
        self.reward_name = reward_name
        self.reward_sum = 0
        self.episode_steps = 0

    def reset(self, initial_state: GameState):
        if self.episode_steps > 0:
            self.logger.record_mean(f"rewards/" + self.reward_name, self.reward_sum / self.episode_steps)
            self.reward_sum = 0
            self.episode_steps = 0

        self.reward_function.reset(initial_state)

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        rew = self.reward_function.get_reward(player, state, previous_action)
        if player.team_num == common_values.BLUE_TEAM:
            self.reward_sum += rew
            self.episode_steps += 1
        return rew

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        rew = self.reward_function.get_final_reward(player, state, previous_action)
        if player.team_num == common_values.BLUE_TEAM:
            self.reward_sum += rew
            self.episode_steps += 1
        return rew
