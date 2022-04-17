from typing import Union, List, Callable

from torch import nn

from .necto_perceiver import NectoPerceiverNet


class ACAttentionNet(nn.Module):
    def __init__(self,
                 net_arch,
                 network_classes: Union[nn.Module, List[nn.Module]] = NectoPerceiverNet,
                 action_stack_size=1,
                 graph_obs=False):
        super(ACAttentionNet, self).__init__()

        if type(network_classes) is not tuple:
            network_classes = [network_classes] * 2

        self.actor = network_classes[0](**net_arch[0])
        self.critic = network_classes[1](**net_arch[1])

        self.action_stack_size = action_stack_size
        self.graph_obs = graph_obs
        self.extract_features: Callable = (self._extract_graph_features if graph_obs
                                           else self._extract_features)

        # Adding required latent dims
        self.latent_dim_pi = self.actor.latent_dims
        self.latent_dim_vf = self.critic.latent_dims

    def _extract_features(self, features):
        """
        :return: query, obs, key padding mask
        """
        # query is first item, the rest are key/value
        # 9 last elements are:
        # -(1 + 8 * action_stack_size):-1 previous actions
        # -1 mask info
        return (features[:, [0], :-1],
                features[:, 1:, :-(1 + 8 * self.action_stack_size)],
                features[:, 1:, -1])

    def _extract_graph_features(self, features):
        """
        :return: query, obs, key padding mask, query edge weights, adjacency matrix
        """
        return (features[:, [0], :-8],
                features[:, 1:, :-(8 + 8 * self.action_stack_size)],
                features[:, 1:, -1],
                features[:, [0], -8:-1],
                features[:, 1:, -8:-1])

    def forward(self, features):
        extracted_features = self.extract_features(features)
        # Squash player dimension to get action distribution
        return (self.actor(*extracted_features).squeeze(1),
                self.critic(*extracted_features).squeeze(1))

    def forward_actor(self, features):
        return self.actor(*self.extract_features(features)).squeeze(1)

    def forward_critic(self, features):
        return self.critic(*self.extract_features(features)).squeeze(1)
