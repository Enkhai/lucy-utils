import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy

from utils.models import ACPerceiverNet


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
