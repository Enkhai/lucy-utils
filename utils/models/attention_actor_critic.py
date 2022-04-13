from typing import Union, List

from torch import nn

from utils.models import NectoPerceiverNet


# TODO: handle graph observation feature extraction
class ACAttentionNet(nn.Module):
    def __init__(self,
                 net_arch,
                 network_classes: Union[nn.Module, List[nn.Module]] = NectoPerceiverNet,
                 action_stack_size=1):
        super(ACAttentionNet, self).__init__()

        if type(network_classes) is not tuple:
            network_classes = [network_classes] * 2

        self.actor = network_classes[0](**net_arch[0])
        self.critic = network_classes[1](**net_arch[1])

        self.action_stack_size = action_stack_size

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

    def forward(self, features):
        query, obs, key_padding_mask = self._extract_features(features)
        # Squash player dimension to get action distribution
        return (self.actor(query, obs, key_padding_mask).squeeze(1),
                self.critic(query, obs, key_padding_mask).squeeze(1))

    def forward_actor(self, features):
        return self.actor(*self._extract_features(features)).squeeze(1)

    def forward_critic(self, features):
        return self.critic(*self._extract_features(features)).squeeze(1)
