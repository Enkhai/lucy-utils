import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn

from utils.models import MLPPolicyNetwork, ACPerceiverNet


class ACMLPPolicy(ActorCriticPolicy):
    """
    Simple ActorCritic policy that uses a custom MLP network
    """

    def __init__(self, *args,
                 nw_hidden_dims=128,
                 nw_n_layers=4,
                 nw_n_heads=4,
                 nw_activation_fn=nn.ReLU,
                 nw_dropout=0.1,
                 **kwargs):
        self.nw_hidden_dims = nw_hidden_dims
        self.nw_n_layers = nw_n_layers
        self.nw_n_heads = nw_n_heads
        self.nw_activation_fn = nw_activation_fn
        self.nw_dropout = nw_dropout

        super(ACMLPPolicy, self).__init__(*args, **kwargs)
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = MLPPolicyNetwork(self.observation_space.shape[-1],
                                              self.nw_hidden_dims,
                                              self.nw_n_layers,
                                              self.nw_activation_fn,
                                              self.nw_dropout)


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
