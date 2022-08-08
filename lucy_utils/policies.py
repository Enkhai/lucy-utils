from functools import partial
from typing import Union, List

import numpy as np
import torch as th
from stable_baselines3.common.distributions import DiagGaussianDistribution, StateDependentNoiseDistribution, \
    MultiCategoricalDistribution, CategoricalDistribution, BernoulliDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
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


class AuxACAttnPolicy(ActorCriticAttnPolicy):
    # TODO: Import RP and SR architectures and fix this
    rp_arch_map = {"base": None,
                   "deep": None,
                   "seq": None}
    sr_arch_map = {"base": None,
                   "deep": None,
                   "deep2": None}

    def __init__(self,
                 *args,
                 use_rp: bool = False,
                 use_sr: bool = False,
                 rp_arch: str = "deep",  # base | deep | seq
                 sr_arch: str = "deep2",  # base | deep | deep2
                 rp_seq_len: int = 20,
                 **kwargs):
        assert rp_arch in self.rp_arch_map, "`rp_arch` can only be `base`, `deep` or `seq`"
        assert sr_arch in self.sr_arch_map, "`sr_arch` can only be `base`, `deep` or `deep2`"

        self.use_rp = use_rp
        self.use_sr = use_sr
        self.rp_arch = rp_arch
        self.sr_arch = sr_arch
        self.rp_seq_len = rp_seq_len

        super(AuxACAttnPolicy, self).__init__(*args, **kwargs)

    def forward_rp(self, obs: th.Tensor) -> th.Tensor:
        """
        AUX: Forward pass in reward prediction network

        :param obs: observation sequence batch
        :return: one-hot-encoded output
        """
        return self.rp_net(obs)

    def forward_sr(self, obs: th.Tensor) -> th.Tensor:
        """
        AUX: Forward pass in state representation network

        :param obs: observation batch
        :return: obs-shaped output
        """
        return self.sr_net(obs)

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """

        # mlp extractor initialization, builds actor and critic networks and their optional shared part
        self._build_mlp_extractor()

        # AUX nets initialization
        shared_net_out = self.mlp_extractor.shared_net(self.features_extractor(th.rand(1, self.features_dim))).shape

        if self.use_rp:
            rp_class = self.rp_arch_map[self.rp_arch]
            self.rp_net = rp_class(self.features_extractor,
                                   self.mlp_extractor.shared_net,
                                   shared_net_out,
                                   seq_len=self.rp_seq_len,
                                   device=self.device)
        if self.use_sr:
            sr_class = self.sr_arch_map[self.sr_arch]
            self.sr_net = sr_class(self.features_extractor,
                                   self.mlp_extractor.shared_net,
                                   shared_net_out,
                                   obs_shape=self.observation_space.shape[0],
                                   device=self.device)

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, latent_sde_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist,
                        (CategoricalDistribution, MultiCategoricalDistribution, BernoulliDistribution)):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
