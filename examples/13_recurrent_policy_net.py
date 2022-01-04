from collections import deque

import rlgym
import torch as th
from rlgym.envs.match import Match
from rlgym.utils.reward_functions import common_rewards
from rlgym.utils.state_setters import DefaultState
from rlgym.utils.terminal_conditions import common_conditions
from rlgym_tools.extra_action_parsers.kbm_act import KBMAction
from rlgym_tools.extra_rewards.diff_reward import DiffReward
from rlgym_tools.sb3_utils.sb3_log_reward import SB3CombinedLogReward, SB3CombinedLogRewardCallback
from rlgym_tools.sb3_utils.sb3_multiple_instance_env import SB3MultipleInstanceEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from utils.obs import SimpleObs

# device = "cuda:0" if th.cuda.is_available() else "cpu"
device = "cpu"


class LSTMFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim, n_steps=32):
        super(LSTMFeaturesExtractor, self).__init__(observation_space, features_dim)
        self.n_steps = n_steps
        # TODO: fix default obs
        self.default_obs: th.Tensor = th.zeros(observation_space.shape[-1], device=device)
        self._reset_steps()

    def _reset_steps(self) -> None:
        self.steps = deque([self.default_obs.clone() for _ in range(self.n_steps)], maxlen=self.n_steps)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # TODO: handle env reset
        # In case of observation batch return a batch of rolling sequences
        if observations.shape[0] > 1:
            # TODO: Prepending default observation is not good, batch observations have their own
            #  previous observations as well. Can I handle this with a longer deque?
            # Prepend default observation to observations
            observations = th.concat([th.stack([self.default_obs.clone() for _ in range(self.n_steps - 1)]),
                                      observations])
            # Stack the observations into fixed length sequences
            return th.stack([observations[i:i + self.n_steps] for i in range(len(observations) - self.n_steps + 1)])
        else:
            self.steps.append(observations.squeeze(0))
            return th.stack(list(self.steps)) \
                .unsqueeze(0) \
                .to(device)


# Rocket League approaches Markovian as the number of players decrease and there is no hidden
# information in the game state
# Due to this, recurrent networks may not be very efficient in this case and LSTMs, in particular,
# are very computationally heavy, hence not ideal for solving Rocket League
class LSTMNetwork(nn.Module):
    def __init__(self, n_features, hidden_size=512, num_layers=1):
        super(LSTMNetwork, self).__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # torch.nn.RNNBase arguments
        self.lstm = nn.LSTM(input_size=n_features,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.latent_dim_pi, self.latent_dim_vf = (hidden_size,) * 2

    def _init_hidden(self):
        return th.zeros((self.num_layers, 2, self.hidden_size), device=device) \
            .requires_grad_() \
            .split(1, 1)

    def forward(self, features):
        out, _ = self.lstm(features, self._init_hidden())

        return out[:, -1, :], out[:, -1, :]


class ACLSTMPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(ACLSTMPolicy, self).__init__(*args, **kwargs)
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = LSTMNetwork(self.features_dim,
                                         self.net_arch[0]['hidden_size'],
                                         self.net_arch[0]['num_layers'])


reward = SB3CombinedLogReward.from_zipped(
    (DiffReward(common_rewards.LiuDistancePlayerToBallReward()), 0.05),
    (DiffReward(common_rewards.LiuDistanceBallToGoalReward()), 10),
    (common_rewards.ConstantReward(), -0.01),
    (common_rewards.EventReward(touch=0.05, goal=10)),
)
reward_names = ["PlayerToBallDistDiff", "BallToGoalDistDiff", "ConstantNegative", "GoalOrTouch"]
models_folder = "models/"


def get_match():
    return Match(reward_function=reward,
                 terminal_conditions=[common_conditions.TimeoutCondition(500),
                                      common_conditions.GoalScoredCondition()],
                 obs_builder=SimpleObs(flatten=True),
                 state_setter=DefaultState(),
                 action_parser=KBMAction(),
                 game_speed=500)


if __name__ == '__main__':
    # env = SB3MultipleInstanceEnv(match_func_or_matches=get_match,
    #                              num_instances=3,
    #                              wait_time=20)
    env = rlgym.make(reward_fn=reward,
                     terminal_conditions=[common_conditions.TimeoutCondition(500),
                                          common_conditions.GoalScoredCondition()],
                     obs_builder=SimpleObs(flatten=True),
                     state_setter=DefaultState(),
                     action_parser=KBMAction(),
                     game_speed=500)

    policy_kwargs = dict(features_extractor_class=LSTMFeaturesExtractor,
                         features_extractor_kwargs=dict(features_dim=env.observation_space.shape[-1], n_steps=32),
                         net_arch=[dict(hidden_size=512,
                                        num_layers=2)],
                         )
    model = PPO(policy=ACLSTMPolicy,
                env=env,
                learning_rate=1e-4,
                tensorboard_log="./bin",
                policy_kwargs=policy_kwargs,
                verbose=1,
                device=device,
                )

    callbacks = [SB3CombinedLogRewardCallback(reward_names),
                 CheckpointCallback(model.n_steps * 100,
                                    save_path=models_folder + "LSTM1_1x512",
                                    name_prefix="model")]
    model.learn(total_timesteps=100_000_000, callback=callbacks, tb_log_name="PPO_LSTM1_1x512")
    model.save(models_folder + "LSTM1_1x512_final")

    env.close()
