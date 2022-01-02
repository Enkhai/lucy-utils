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
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn


class SimpleObs(ObsBuilder):
    """
    Simple observation builder for a ball and one car only
    Observation space is of shape 1 * 2 * 20:
    1 (batch)
    * 2 (1 ball + 1 car)
    * 20 (2 (car and ball flags)
        + 9 ((relative) standardized position, linear velocity and angular velocity  3-d vectors)
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


# We build yet another custom policy and value network, just as an example
# A better version of an MLP extractor can be found in `stable_baselines3.common.torch_layers.MlpExtractor`,
# as well as `stable_baselines3.common.torch_layers.create_mlp`
class MLPNetwork(nn.Module):
    def __init__(self, n_features, hidden_dims, n_layers, activation_fn=nn.ReLU, dropout=0.1, output_features=None):
        super(MLPNetwork, self).__init__()
        # First layer
        m = (nn.Linear(n_features, hidden_dims),
             activation_fn(),
             nn.Dropout(dropout))
        # Hidden layers
        m += (nn.Linear(hidden_dims, hidden_dims),
              activation_fn(),
              nn.Dropout(dropout),) * (n_layers - 2)
        # Final layer
        # If the final layer is the output, simply add a linear layer
        if output_features:
            m += (nn.Linear(hidden_dims, output_features),)
        # If it's not, also add the activation function and another dropout
        else:
            m += (nn.Linear(hidden_dims, hidden_dims), activation_fn(), nn.Dropout(dropout))
        self.model = nn.Sequential(*m)

        self.latent_dim_pi, self.latent_dim_vf = (hidden_dims,) * 2

    def forward(self, features):
        # Value and policy share the shame architecture
        return self.model(features), self.model(features)


# We use this network with a custom AC policy
class ACMLPPolicy(ActorCriticPolicy):

    def __init__(self, *args,
                 net_hidden_dims=256,
                 net_n_layers=5,
                 net_n_heads=4,
                 net_activation_fn=nn.ReLU,
                 net_dropout=0.1,
                 **kwargs):
        self.net_hidden_dims = net_hidden_dims
        self.net_n_layers = net_n_layers
        self.net_n_heads = net_n_heads
        self.net_activation_fn = net_activation_fn
        self.net_dropout = net_dropout

        super(ACMLPPolicy, self).__init__(*args, **kwargs)
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = MLPNetwork(self.observation_space.shape[-1],
                                        self.net_hidden_dims,
                                        self.net_n_layers,
                                        self.net_activation_fn,
                                        self.net_dropout)


if __name__ == '__main__':
    # This is quite a sparse reward that may aid our player to learn and shoot the ball towards the goal
    # We don't want to punish our network a lot, since it's a rather simple network and doing so may hinder learning
    reward = SB3CombinedLogReward.from_zipped(
        (DiffReward(common_rewards.LiuDistancePlayerToBallReward()), 0.05),
        (DiffReward(common_rewards.LiuDistanceBallToGoalReward()), 10),
        (common_rewards.ConstantReward(), -0.03),
        (common_rewards.EventReward(touch=0.05, goal=10)),
    )
    reward_names = ["PlayerToBallDistDiff", "BallToGoalDistDiff", "ConstantNegative", "GoalOrTouch"]

    env = rlgym.make(game_speed=500,
                     terminal_conditions=[TimeoutCondition(500), GoalScoredCondition()],
                     reward_fn=reward,
                     obs_builder=SimpleObs(flatten=True),
                     action_parser=KBMAction())  # We use a KeyBoardMouse action parser this time around

    model = PPO(policy=ACMLPPolicy,
                env=env,
                tensorboard_log="./bin",
                verbose=1,
                device="cpu"
                )
    model.set_random_seed(0)

    callbacks = [SB3CombinedLogRewardCallback(reward_names),
                 CheckpointCallback(model.n_steps * 10,
                                    save_path="models",
                                    name_prefix="model")]  # save every 10 rollouts
    model.learn(total_timesteps=100_000_000, callback=callbacks, tb_log_name="PPO_MLP_4x256")
    model.save("model_final")

    env.close()
