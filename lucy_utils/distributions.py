from stable_baselines3.common.distributions import MultiCategoricalDistribution
from torch import nn


class MultiCategoricalDistributionIdentityNet(MultiCategoricalDistribution):

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        return nn.Identity()
