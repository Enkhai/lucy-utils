from typing import Tuple

import rlgym
import torch as th
from rlgym.utils.reward_functions import common_rewards
from rlgym.utils.terminal_conditions.common_conditions import GoalScoredCondition, TimeoutCondition
from rlgym_tools.sb3_utils.sb3_log_reward import SB3CombinedLogReward, SB3CombinedLogRewardCallback
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn


# A custom policy and value network architecture example
class CustomExtractorNetwork(nn.Module):

    def __init__(self, feature_dim: int):
        super(CustomExtractorNetwork, self).__init__()

        self.shared_net = nn.Sequential(nn.Conv1d(1, 3, 3),
                                        nn.ReLU(),
                                        nn.MaxPool1d(3, 2),
                                        nn.Conv1d(3, 8, 3),
                                        nn.MaxPool1d(3, 2),
                                        nn.ReLU())

        self.policy_net = nn.Sequential(self.shared_net,
                                        nn.Conv1d(8, 16, 3),
                                        nn.ReLU(),
                                        nn.MaxPool1d(3, 2),
                                        nn.Flatten())
        self.value_net = nn.Sequential(self.shared_net,
                                       nn.Conv1d(8, 16, 3),
                                       nn.ReLU(),
                                       nn.MaxPool1d(3, 2),
                                       nn.Flatten())

        # Get the number of output features
        # *We don't compute gradients*
        with th.no_grad():
            # Batch size 1 * 1 channel * observation features
            # This is necessary for the observation space vector to be passed through the network
            random_features: th.Tensor = th.rand(feature_dim).unsqueeze(0).unsqueeze(0)
            # Policy and value network number of features will be exactly the same since they share
            # the same architecture
            policy_net_n_feat = self.value_net(random_features).shape[1]
            value_net_n_feat = self.policy_net(random_features).shape[1]

        # Subclassing the ActorCriticPolicy further on in Stable Baselines 3 always uses a linear layer
        # after the policy and value networks to match the dimensions of the action space
        # The `latent_dim` variables are used for building those exact linear layers
        # Those are the number of features we computed earlier
        self.latent_dim_pi = policy_net_n_feat
        self.latent_dim_vf = value_net_n_feat

    # We always need two outputs in Actor-Critic algorithms:
    # one for the policy and one for the value function
    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        # Add channel dimension for the CNN forward pass
        reshaped_features = features.unsqueeze(1)
        return self.policy_net(reshaped_features), self.value_net(reshaped_features)


# A custom policy example
# For PPO, typing suggests that the policy must be an ActorCriticPolicy
class CustomActorCriticPolicy(ActorCriticPolicy):

    # ActorCriticPolicy arguments
    # You can cherry-pick some of those for initialization
    def __init__(self, *args, **kwargs):
        super(CustomActorCriticPolicy, self).__init__(*args, **kwargs)

        # Disable orthogonal initialization: This is useful for combating the problem of
        # vanishing and exploding gradients in deep neural networks
        # https://smerity.com/articles/2016/orthogonal_init.html
        self.ortho_init = False

    # This is used for building the architecture responsible for handling the policy and value networks
    # The `mpl_extractor` is a convention used to name this architecture
    def _build_mlp_extractor(self) -> None:
        # We simply use the custom policy network we built above
        self.mlp_extractor = CustomExtractorNetwork(self.features_dim)


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

    model = PPO(policy=CustomActorCriticPolicy,
                env=env,
                verbose=1,
                device="cpu")
    model.set_random_seed(0)

    reward_log_callback = SB3CombinedLogRewardCallback(rew_names=reward_names)
    model.learn(total_timesteps=100_000_000, callback=reward_log_callback)

    env.close()
