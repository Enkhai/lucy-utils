from stable_baselines3.common.distributions import CategoricalDistribution
from torch import nn


class CategoricalDistributionIdentityNet(CategoricalDistribution):

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        return nn.Identity()
