import warnings
from functools import partial
from typing import Union, List, Optional, Type, Dict, Any

import gym
import numpy as np
import torch as th
from stable_baselines3.common.distributions import (DiagGaussianDistribution,
                                                    StateDependentNoiseDistribution,
                                                    MultiCategoricalDistribution,
                                                    CategoricalDistribution,
                                                    BernoulliDistribution, make_proba_distribution)
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor, NatureCNN
from stable_baselines3.common.type_aliases import Schedule
from torch import nn

from .distributions import CategoricalDistributionIdentityNet
from .models import (ACAttentionNet,
                     NectoPerceiverNet,
                     SeqRewardPredictionNetwork,
                     DeepStateRepresentationNetwork)


class ActorCriticAttnPolicy(ActorCriticPolicy):
    def __init__(self,
                 observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 lr_schedule: Schedule,
                 net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
                 activation_fn: Type[nn.Module] = nn.Tanh,
                 ortho_init: bool = True,
                 use_sde: bool = False,
                 log_std_init: float = 0.0,
                 full_std: bool = True,
                 sde_net_arch: Optional[List[int]] = None,
                 use_expln: bool = False,
                 squash_output: bool = False,
                 features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
                 features_extractor_kwargs: Optional[Dict[str, Any]] = None,
                 normalize_images: bool = True,
                 optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
                 optimizer_kwargs: Optional[Dict[str, Any]] = None,
                 # Custom args
                 network_classes: Union[nn.Module, List[nn.Module]] = NectoPerceiverNet,
                 action_stack_size: int = 1,
                 graph_obs: bool = False,
                 is_nexto: bool = False,
                 ):

        # +++ START custom code +++
        self.action_stack_size = action_stack_size
        self.network_classes = network_classes
        self.graph_obs = graph_obs
        self.is_nexto = is_nexto
        # +++ END custom code +++

        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        super(ActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output,
        )

        # Default network architecture, from stable-baselines
        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = []
            else:
                net_arch = [dict(pi=[64, 64], vf=[64, 64])]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.ortho_init = ortho_init

        self.features_extractor = features_extractor_class(self.observation_space, **self.features_extractor_kwargs)
        self.features_dim = self.features_extractor.features_dim

        self.normalize_images = normalize_images
        self.log_std_init = log_std_init
        dist_kwargs = None
        # Keyword arguments for gSDE distribution
        if use_sde:
            dist_kwargs = {
                "full_std": full_std,
                "squash_output": squash_output,
                "use_expln": use_expln,
                "learn_features": False,
            }

        if sde_net_arch is not None:
            warnings.warn("sde_net_arch is deprecated and will be removed in SB3 v2.4.0.", DeprecationWarning)

        self.use_sde = use_sde
        self.dist_kwargs = dist_kwargs

        # +++ Custom action distribution +++
        if is_nexto:
            self.action_dist = CategoricalDistributionIdentityNet(action_space.n, **dist_kwargs)
        else:
            self.action_dist = make_proba_distribution(action_space, use_sde=use_sde, dist_kwargs=dist_kwargs)

        self._build(lr_schedule)

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
    def __init__(self,
                 *args,
                 use_rp: bool = False,
                 use_sr: bool = False,
                 n_aux_heads: Union[int, List[int]] = 4,
                 rp_seq_len: int = 20,
                 **kwargs):

        self.use_rp = use_rp
        self.use_sr = use_sr
        if type(n_aux_heads) is not list:
            self.n_aux_heads = [n_aux_heads] * 2
        else:
            self.n_aux_heads = n_aux_heads
        self.rp_seq_len = rp_seq_len

        super(AuxACAttnPolicy, self).__init__(*args, **kwargs)

    def forward_rp(self, obs: th.Tensor) -> th.Tensor:
        """
        AUX: Forward pass in reward prediction network

        :param obs: Observation sequence batch of shape (batch_size, seq_len) + obs_shape
        :return: 3-class regression output
        """
        batch_size = obs.shape[0]
        # batch, sequence, n_obj, n_features -> batch * sequence, n_obj, n_features
        obs_squeezed = obs.view((batch_size * self.rp_seq_len,) + obs.shape[2:])

        return self.rp_net(batch_size, *self.mlp_extractor.extract_features(obs_squeezed))

    def forward_sr(self, obs: th.Tensor) -> th.Tensor:
        """
        AUX: Forward pass in state representation network

        :param obs: Observation batch
        :return: flat key/value observation-shaped output
        """
        return self.sr_net(*self.mlp_extractor.extract_features(obs))

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """

        # mlp extractor initialization, builds actor and critic networks and their optional shared part
        self._build_mlp_extractor()

        if self.use_rp:
            self.rp_net = SeqRewardPredictionNetwork(self.mlp_extractor.actor,  # custom Perceiver arch
                                                     self.rp_seq_len,
                                                     self.n_aux_heads[0],
                                                     self.device)
        if self.use_sr:
            # flat kv obs
            obs_shape = (self.observation_space.shape[0] - 1) * self.net_arch[0]["kv_dims"]
            self.sr_net = DeepStateRepresentationNetwork(self.mlp_extractor.actor,  # custom Perceiver arch
                                                         obs_shape,
                                                         self.n_aux_heads[1],
                                                         self.device)

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
