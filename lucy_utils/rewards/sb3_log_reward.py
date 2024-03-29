import os

import numpy as np
from rlgym.utils import RewardFunction, common_values
from rlgym.utils.gamestates import PlayerData, GameState
from stable_baselines3.common.callbacks import BaseCallback


class SB3NamedLogReward(RewardFunction):
    def __init__(self,
                 reward_function: RewardFunction,
                 reward_name: str,
                 prefix: str = "rewards",
                 blue_only: bool = True,
                 folder_location: str = 'combinedlogfiles',
                 log: bool = True
                 ):
        """
        Reward function wrapper for logging individual mean episode rewards to a logfile.

        The logfile location is formatted as such:
        `<folder_location>/<prefix>/<reward_name>`. The reward function works in conjunction with
        `SB3NamedLogRewardCallback` for logging mean episode rewards into TensorBoard.

        The logfiles are deleted after the reward object is destroyed.
        Always make sure that the combination of the reward name
        and the prefix are unique to each instance of the `SB3NamedLogReward`.

        Logging for the blue team only is enabled by default, since blue and orange rewards can often be symmetric and
        add up to 0.

        :param reward_function: RLGym `RewardFunction` to log
        :param reward_name: String name used for logging the reward value
        :param prefix: The prefix to use for logging, set to `rewards` by default
        :param blue_only: Dictates whether to log blue team rewards only, set to `True` by default
        :param folder_location: The upper level folder location into which to log the rewards, set to `combinedlogfiles`
         by default
        :param log: Dictates whether to log the reward, set to `True` by default
        """

        super(SB3NamedLogReward, self).__init__()
        self.reward_function = reward_function
        self.reward_name = reward_name
        self.log = log

        # Logging enabled
        if log:
            self.blue_only = blue_only
            self.file_location = folder_location + "/" + prefix + "/" + reward_name
            self.reward_sum = 0
            self.episode_steps = 0

            # Make sure there is a folder to dump to
            os.makedirs(folder_location + "/" + prefix, exist_ok=True)

            try:
                # Create the log file by opening in x mode
                with open(self.file_location, 'x'):
                    pass
            except FileExistsError:
                # If the file exists, empty it
                with open(self.file_location, 'w'):
                    pass

    def __del__(self):
        if self.log:
            # On delete, remove the logfile
            os.remove(self.file_location)

    def reset(self, initial_state: GameState):
        if self.log:
            self.reward_sum = 0
            self.episode_steps = 0

        self.reward_function.reset(initial_state)

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        rew = self.reward_function.get_reward(player, state, previous_action)
        if self.log:
            if (self.blue_only and player.team_num == common_values.BLUE_TEAM) or not self.blue_only:
                self.reward_sum += rew
                self.episode_steps += 1
        return rew

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        rew = self.reward_function.get_final_reward(player, state, previous_action)
        if self.log:
            if (self.blue_only and player.team_num == common_values.BLUE_TEAM) or not self.blue_only:
                self.reward_sum += rew
                self.episode_steps += 1
                # reward mean is exact for the last player only
                with open(self.file_location, 'a') as f:
                    f.write(str(self.reward_sum / self.episode_steps) + '\n')
        return rew


class SB3NamedLogRewardCallback(BaseCallback):
    """
    SB3 reward log callback to be used in conjunction with the `SB3NamedLogReward` for
    logging episode reward values into TensorBoard.
    """

    def __init__(self, logger_idx: int = 0, folder_location: str = 'combinedlogfiles'):
        """
        :param logger_idx: The environment index for which to log rewards. Since RLGym treats each player as a
         separate environment, setting, for example, logger_idx=3 verifies whether the episode has ended for player
         number 4, in order to log reward values.
        :param folder_location: The upper level folder into which rewards are logged.
        """
        super(SB3NamedLogRewardCallback, self).__init__()
        self.folder_location = folder_location
        self.logger_idx = logger_idx

    def _on_step(self) -> bool:
        if self.locals['dones'][self.logger_idx]:
            # Read folder and log reward
            for upper in os.listdir(self.folder_location):  # `rewards` / `utility` / etc.
                upper_f_name = self.folder_location + "/" + upper + "/"
                for lower in os.listdir(upper_f_name):  # <reward_name>
                    f_name = upper_f_name + lower
                    with open(f_name, "r") as f:
                        cont = f.readlines()
                        # If a reward has been logged
                        if cont:
                            # Send the last value to TensorBoard
                            self.model.logger.record(upper + "/" + lower, float(cont[-1].strip('\n')))
                            # and empty the dumpfile
                            with open(f_name, "w"):
                                pass

            self.model.logger.dump(self.num_timesteps)
        return True
