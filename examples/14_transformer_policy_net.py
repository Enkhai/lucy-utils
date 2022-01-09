import random
from typing import Any

import numpy as np
import torch as th
from rlgym.envs.match import Match
from rlgym.utils import ObsBuilder
from rlgym.utils import common_values
from rlgym.utils.gamestates import PlayerData, GameState
from rlgym.utils.reward_functions import common_rewards
from rlgym.utils.state_setters import DefaultState
from rlgym.utils.terminal_conditions import common_conditions
from rlgym_tools.extra_action_parsers.kbm_act import KBMAction
from rlgym_tools.extra_rewards.diff_reward import DiffReward
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv
from rlgym_tools.sb3_utils.sb3_log_reward import SB3CombinedLogReward, SB3CombinedLogRewardCallback
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import create_mlp
from torch import nn

# device = "cuda:0" if th.cuda.is_available() else "cpu"
device = "cpu"


class AttentionObs(ObsBuilder):
    """
    Observation builder suitable for attention models\n
    Returns an observation tensor for a player:\n
    1 (batch)\n
    \* 8: 1 (player) + 1 (ball) + number of players (default is 6)\n
    \* 32: 4 (player flag, blue team flag, orange team flag, ball flag) + 9 (normalized position, normalized linear
    velocity and normalized angular velocity vectors) + 6 (forward and upward rotation axes) + 4 (boost amount,
    touching ground flag, has flip flag, is demoed flag) + 8 (previous action) + 1 (key padding mask
    used for marking object/player padding in observation)
    """

    current_state = None
    current_obs = None

    # Inversion vector - needed for invariance, used to invert the x and y axes for the orange team observation
    _invert = np.array([1] * 4 + [-1, -1, 1] * 5 + [1] * 4 + [1] * 9)

    def __init__(self, n_players=6):
        super(AttentionObs, self).__init__()
        self.n_players = n_players

    def reset(self, initial_state: GameState):
        pass

    def _update_state_and_obs(self, state: GameState):
        obs = np.zeros((1 + self.n_players, 23 + 8 + 1))
        obs[:, -1] = 1  # key padding mask

        # Ball
        ball = state.ball
        obs[0, 3] = 1  # ball flag
        obs[0, 4:7] = ball.position / common_values.CAR_MAX_SPEED
        obs[0, 7:10] = ball.linear_velocity / common_values.CAR_MAX_SPEED
        obs[0, 10:13] = ball.angular_velocity / common_values.CAR_MAX_ANG_VEL
        # no forward, upward, boost amount, touching ground and flip info for ball
        obs[0, -1] = 0  # mark non-padded

        # Players
        for i, p in zip(range(1, len(state.players) + 1), state.players):
            if p.team_num == common_values.BLUE_TEAM:  # team flags
                obs[i, 1] = 1
            else:
                obs[i, 2] = 1
            p_car = p.car_data
            obs[i, 4:7] = p_car.position / common_values.CAR_MAX_SPEED
            obs[i, 7:10] = p_car.linear_velocity / common_values.CAR_MAX_SPEED
            obs[i, 10:13] = p_car.angular_velocity / common_values.CAR_MAX_ANG_VEL
            obs[i, 13:16] = p_car.forward()
            obs[i, 16:19] = p_car.up()
            # we could also use p_car.right(), steering might be useful
            obs[i, 19] = p.boost_amount
            obs[i, 20] = p.on_ground
            obs[i, 21] = p.has_flip
            obs[i, 22] = p.is_demoed
            obs[i, -1] = 0  # mark non-padded

        self.current_obs = obs
        self.current_state = state

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:
        # No need to update the state until model produces output for all cars
        # When it does and the state changes, update
        if state != self.current_state:
            self._update_state_and_obs(state)

        obs = self.current_obs.copy()

        player_idx = state.players.index(player) + 1  # plus one because ball is first
        obs[player_idx, 0] = 1  # player flag
        if player.team_num == common_values.ORANGE_TEAM:  # if orange team
            obs[:, [1, 2]] = obs[:, [2, 1]]  # swap team flags
            obs *= self._invert  # invert x and y axes

        query = obs[[player_idx], :]
        query[0, -9:-1] = previous_action  # add previous action to query player

        obs[:, 4:10] -= query[0, 4:10]  # relative position and linear velocity
        obs = np.concatenate([query, obs])

        return obs


# Inspired by Necto's EARL model and Perceiver https://arxiv.org/abs/2103.03206
class PerceiverNet(nn.Module):
    class PerceiverBlock(nn.Module):
        def __init__(self, d_model, ca_nhead, latent_nhead):
            super().__init__()
            # TODO: replace this with something else. The latent transformer is probably not very good.
            #  Necto doesn't work like that. Need to check for dropout activation during
            #  the actor pass (model.training), maybe not good. Also, batch norm bad. CPU is still(!) faster than GPU,
            #  even with this setup(!)
            self.cross_attention = nn.MultiheadAttention(d_model, ca_nhead, batch_first=True)
            self.latent_transformer = nn.TransformerEncoderLayer(d_model, latent_nhead, d_model, batch_first=True)
            self._reset_parameters()

        # Following PyTorch Transformer implementation
        def _reset_parameters(self):
            r"""
            Initiate parameters in the transformer model.
            """
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        def forward(self, latent, byte, key_padding_mask=None):
            ca_out = self.cross_attention(latent, byte, byte, key_padding_mask)[0]  # attention outputs only
            return self.latent_transformer(ca_out)

    def __init__(self, net_arch):
        super(PerceiverNet, self).__init__()

        # Parse arguments
        self.query_dims = net_arch["query_dims"]
        self.kv_dims = net_arch["kv_dims"]
        self.hidden_dims = net_arch["hidden_dims"] if "hidden_dims" in net_arch else 256
        self.n_layers = net_arch["n_layers"] if "n_layers" in net_arch else 4
        self.ca_nhead = net_arch["n_ca_heads"] if "n_ca_heads" in net_arch else 1
        self.latent_nhead = net_arch["n_latent_heads"] if "n_latent_heads" in net_arch else 4
        if "n_preprocess_layers" in net_arch and net_arch["n_preprocess_layers"] > 1:
            self.n_preprocess_layers = net_arch["n_preprocess_layers"]
        else:
            self.n_preprocess_layers = net_arch["n_preprocess_layers"] = 1

        if "n_postprocess_layers" in net_arch and net_arch["n_postprocess_layers"] > 0:
            self.n_postprocess_layers = net_arch["n_postprocess_layers"]
        else:
            self.n_postprocess_layers = net_arch["n_postprocess_layers"] = 0

        # Build the architecture
        self.query_preprocess = nn.Sequential(*create_mlp(self.query_dims,
                                                          -1,
                                                          [self.hidden_dims] * self.n_preprocess_layers))
        self.kv_preprocess = nn.Sequential(*create_mlp(self.kv_dims,
                                                       -1,
                                                       [self.hidden_dims] * self.n_preprocess_layers))
        # If the network is recurrent repeat the same block at each layer
        if "recurrent" in net_arch and net_arch["recurrent"]:
            self.perceiver_blocks = nn.ModuleList([self.PerceiverBlock(self.hidden_dims,
                                                                       self.ca_nhead,
                                                                       self.latent_nhead)] * self.n_layers)
        # Otherwise, create new blocks for each layer
        else:
            self.perceiver_blocks = nn.ModuleList([self.PerceiverBlock(self.hidden_dims,
                                                                       self.ca_nhead,
                                                                       self.latent_nhead)
                                                   for _ in range(self.n_layers)])

        if self.n_postprocess_layers > 0:
            self.postprocess = nn.Sequential(*create_mlp(self.hidden_dims,
                                                         self.hidden_dims,
                                                         [self.hidden_dims] * (self.n_preprocess_layers - 1)))
        else:
            self.postprocess = nn.Identity()

    def forward(self, query, obs, key_padding_mask=None):
        q_emb = self.query_preprocess(query)
        kv_emb = self.kv_preprocess(obs)

        for block in self.perceiver_blocks:
            q_emb = block(q_emb, kv_emb, key_padding_mask)  # update latent only

        q_emb = self.postprocess(q_emb)
        return q_emb


class ACPerceiverNet(nn.Module):
    def __init__(self, net_arch):
        super(ACPerceiverNet, self).__init__()

        self.actor = PerceiverNet(net_arch[0])
        self.critic = PerceiverNet(net_arch[1])

        # Adding required latent dims
        self.latent_dim_pi = self.actor.hidden_dims
        self.latent_dim_vf = self.critic.hidden_dims

    def forward(self, features):
        key_padding_mask = features[:, 1:, -1]  # first item in the sequence is the query
        query = features[:, [0], :-1]  # we don't need mask info
        obs = features[:, 1:, :-9]  # minus previous actions for obs
        # Squash player dimension to pass latent through SB3 default final linear layer
        return (self.actor(query, obs, key_padding_mask).squeeze(1),
                self.critic(query, obs, key_padding_mask).squeeze(1))


class ACPerceiverPolicy(ActorCriticPolicy):
    def __init__(self, *args,
                 **kwargs):
        super(ACPerceiverPolicy, self).__init__(*args, **kwargs)
        self.ortho_init = False

    # Bypass observation preprocessing and features extractor
    def extract_features(self, obs: th.Tensor) -> th.Tensor:
        return obs.float()  # Handle Double type tensor error

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = ACPerceiverNet(net_arch=self.net_arch)


reward = SB3CombinedLogReward.from_zipped(
    (DiffReward(common_rewards.LiuDistancePlayerToBallReward()), 0.05),
    (DiffReward(common_rewards.LiuDistanceBallToGoalReward()), 10),
    (common_rewards.ConstantReward(), -0.004),
    (common_rewards.EventReward(touch=0.05, goal=10)),
)
reward_names = ["PlayerToBallDistDiff", "BallToGoalDistDiff", "ConstantNegative", "GoalOrTouch"]
models_folder = "models/"


def get_match():
    return Match(reward_function=reward,
                 terminal_conditions=[common_conditions.TimeoutCondition(500),
                                      common_conditions.GoalScoredCondition()],
                 # The number of n_players in AttentionObs must be the same for all environments
                 obs_builder=AttentionObs(),
                 state_setter=DefaultState(),
                 action_parser=KBMAction(),
                 team_size=random.randint(1, 3),  # arbitrary team size
                 self_play=True,  # 😎
                 game_speed=500)


if __name__ == '__main__':
    env = SB3MultipleInstanceEnv(match_func_or_matches=get_match,
                                 num_instances=2,
                                 wait_time=20)

    policy_kwargs = dict(net_arch=[dict(
        # minus one for the discarded key padding mask
        query_dims=env.observation_space.shape[-1] - 1,
        # minus eight for the previous action
        kv_dims=env.observation_space.shape[-1] - 1 - 8,
        # the rest is default arguments
    )] * 2)  # *2 because actor and critic will share the same architecture
    model = PPO(policy=ACPerceiverPolicy,
                env=env,
                learning_rate=1e-4,
                tensorboard_log="./bin",
                policy_kwargs=policy_kwargs,
                verbose=1,
                device=device,
                )
    callbacks = [SB3CombinedLogRewardCallback(reward_names),
                 CheckpointCallback(model.n_steps * 100,
                                    save_path=models_folder + "Perceiver",
                                    name_prefix="model")]
    # 2 because separate actor and critic branches,
    # 4 because 4 perceiver blocks,
    # 256 because 256 perceiver block hidden dims
    model.learn(total_timesteps=100_000_000, callback=callbacks, tb_log_name="PPO_Perceiver2_4x256")
    model.save(models_folder + "Perceiver_final")

    env.close()
