import rlgym
import torch as th
from rlgym.utils.reward_functions import common_rewards
from rlgym.utils.terminal_conditions.common_conditions import GoalScoredCondition, TimeoutCondition
from rlgym_tools.sb3_utils.sb3_log_reward import SB3CombinedLogReward, SB3CombinedLogRewardCallback
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


# Between the environment observation and the policy architecture there exists another model,
# called the features extractor, used for applying various functions to the observation, before it
# is passed on to the policy network
# Here we build a simple features extractor model for the RLGym 1-d observation space
# One can also build multi-input extractors
# You can read more here:
# https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html#multiple-inputs-and-dictionary-observations
class CustomFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim):
        super(CustomFeaturesExtractor, self).__init__(observation_space, features_dim)
        self.model = nn.Sequential(nn.Conv1d(1, 3, 3),
                                   nn.ReLU(),
                                   nn.Conv1d(3, 6, 3),
                                   nn.ReLU(),
                                   nn.Flatten())
        with th.no_grad():
            n_flatten_feats = self.forward(th.rand(features_dim).unsqueeze(0)).shape[1]
        self.model.add_module("linear", nn.Linear(n_flatten_feats, features_dim))
        self.model.add_module("relu_out", nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.model(observations.unsqueeze(1))


if __name__ == '__main__':
    reward = SB3CombinedLogReward.from_zipped(
        (common_rewards.ConstantReward(), -0.02),
        (common_rewards.EventReward(goal=1, concede=-1), 100),
        (common_rewards.VelocityPlayerToBallReward(), 0.05),
        (common_rewards.VelocityBallToGoalReward(), 0.2),
        (common_rewards.TouchBallReward(), 0.2),
        (common_rewards.VelocityReward(), 0.01),
        (common_rewards.LiuDistanceBallToGoalReward(), 0.25),
        (common_rewards.LiuDistancePlayerToBallReward(), 0.1),
        (common_rewards.AlignBallGoal(), 0.15),
        (common_rewards.FaceBallReward(), 0.1)
    )
    reward_names = [fn.__class__.__name__ for fn in reward.reward_functions]

    env = rlgym.make(game_speed=500,
                     terminal_conditions=[TimeoutCondition(500), GoalScoredCondition()],
                     reward_fn=reward)

    # We pass the features extractor class to the policy arguments, along with the observation space dimension,
    # used for building the features extractor network
    policy_kwargs = dict(features_extractor_class=CustomFeaturesExtractor,
                         features_extractor_kwargs=dict(features_dim=env.observation_space.shape[0]))
    model = PPO(policy="MlpPolicy",
                env=env,
                tensorboard_log="./bin",
                verbose=1,
                policy_kwargs=policy_kwargs,
                device="cpu")
    model.set_random_seed(0)

    reward_log_callback = SB3CombinedLogRewardCallback(reward_names=reward_names)
    model.learn(total_timesteps=100_000_000, callback=reward_log_callback)

    env.close()
