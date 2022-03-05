from typing import Any

import numpy as np
import rlgym
from rlgym.utils import common_values
from rlgym.utils.gamestates import GameState, PlayerData
from rlgym.utils.obs_builders import ObsBuilder
from rlgym.utils.reward_functions import common_rewards
from rlgym.utils.terminal_conditions.common_conditions import (GoalScoredCondition,
                                                               TimeoutCondition)
from rlgym_tools.extra_action_parsers.kbm_act import KBMAction
from rlgym_tools.extra_rewards.diff_reward import DiffReward
from rlgym_tools.sb3_utils.sb3_log_reward import SB3CombinedLogReward, SB3CombinedLogRewardCallback
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from torch import nn


class SimpleObs(ObsBuilder):
    """
    Simple observation builder for a ball and one car only\n
    Observation space is of shape 1 * 2 * 20:\n
    1 (batch)\n
    \* 2 (1 ball + 1 car)\n
    \* 20 (2 (car and ball flags)
    + 9 ((relative) standardized position, linear velocity and angular velocity 3-d vectors)
    + 6 (forward and upward rotation axes 3-d vectors)
    + 3 (boost, touching ground and has flip flags))

    If flatten is true, it simply returns a vector of length 40 (2 * 20)
    """
    POS_STD = 3000

    def __init__(self, flatten: bool = False):
        super(SimpleObs, self).__init__()
        # The `flatten` boolean is useful for MLP networks
        self.flatten = flatten

    def reset(self, initial_state: GameState):
        # build_obs is called automatically after environment reset
        pass

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:
        # We don't consider teams or inverted data, the observation builder
        # is built just for one blue team car and a ball

        ball = state.ball
        car = player.car_data

        ball_obs = np.concatenate([[1, 0],  # 1 for ball
                                   # standardized relative position
                                   (ball.position - car.position) / self.POS_STD,
                                   # standardized relative velocity
                                   (ball.linear_velocity - car.linear_velocity) / common_values.BALL_MAX_SPEED,
                                   # angular velocity not relative, car and ball share the same max angular velocities
                                   ball.angular_velocity / common_values.CAR_MAX_ANG_VEL,
                                   np.zeros(6),  # no rotation axes
                                   np.zeros(3)])  # no boost, touching ground and has flip flags
        car_obs = np.concatenate([[0, 1],  # 1 for car
                                  car.position / self.POS_STD,
                                  car.linear_velocity / common_values.CAR_MAX_SPEED,
                                  car.angular_velocity / common_values.CAR_MAX_ANG_VEL,
                                  car.forward(),
                                  car.up(),
                                  [player.boost_amount, player.on_ground, player.has_flip]])

        # In the case of an MLP policy network, return a concatenated 1-d array
        if self.flatten:
            return np.concatenate([ball_obs, car_obs])
        return np.stack([ball_obs, car_obs])


if __name__ == '__main__':
    # This is quite a sparse reward that may aid our player to learn and shoot the ball towards the goal
    # We don't want to punish our network a lot, since it's a rather simple network and doing so may hinder learning
    reward = SB3CombinedLogReward.from_zipped(
        (DiffReward(common_rewards.LiuDistancePlayerToBallReward()), 0.05),
        (DiffReward(common_rewards.LiuDistanceBallToGoalReward()), 10),
        (common_rewards.ConstantReward(), -0.01),
        (common_rewards.EventReward(touch=0.05, goal=10)),
    )
    reward_names = ["PlayerToBallDistDiff", "BallToGoalDistDiff", "ConstantNegative", "GoalOrTouch"]

    env = rlgym.make(game_speed=500,
                     terminal_conditions=[TimeoutCondition(500), GoalScoredCondition()],
                     reward_fn=reward,
                     obs_builder=SimpleObs(flatten=True),
                     # We use a KeyBoardMouse action parser this time around
                     # By doing so, the 8 Bakkesmod RLGym plugin API actions are produced by 5 action outputs only
                     # The actions in this case are also discrete, meaning they either happen or they don't,
                     # similarly to keyboard outputs
                     action_parser=KBMAction())

    policy_kwargs = dict(net_arch=[256] * 5,
                         activation_fn=nn.ReLU)
    model = PPO(policy="MlpPolicy",
                env=env,
                policy_kwargs=policy_kwargs,
                tensorboard_log="./bin",
                verbose=1,
                device="cpu"
                )
    model.set_random_seed(0)

    models_folder = "models/"
    callbacks = [SB3CombinedLogRewardCallback(reward_names),
                 CheckpointCallback(model.n_steps * 10,  # save every 10 rollouts
                                    save_path=models_folder + "MLP1",
                                    name_prefix="model")]
    model.learn(total_timesteps=100_000_000, callback=callbacks, tb_log_name="PPO_MLP_4x256")
    model.save(models_folder + "model_final")

    env.close()
