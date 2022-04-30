from typing import Union, List

import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn

from .models import ACAttentionNet
from .models import NectoPerceiverNet


class ActorCriticAttnPolicy(ActorCriticPolicy):
    def __init__(self,
                 *args,
                 network_classes: Union[nn.Module, List[nn.Module]] = NectoPerceiverNet,
                 action_stack_size: int = 1,
                 graph_obs: bool = False,
                 **kwargs):
        self.action_stack_size = action_stack_size
        self.network_classes = network_classes
        self.graph_obs = graph_obs
        super(ActorCriticAttnPolicy, self).__init__(*args, **kwargs)

    # Bypass observation preprocessing and features extractor
    def extract_features(self, obs: th.Tensor) -> th.Tensor:
        return obs.float()  # Handle Double type tensor error

    def _build_mlp_extractor(self) -> None:
        # self.ortho_init = False
        self.mlp_extractor = ACAttentionNet(self.net_arch,
                                            self.network_classes,
                                            self.action_stack_size,
                                            self.graph_obs)
