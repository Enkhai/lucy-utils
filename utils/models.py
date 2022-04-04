from stable_baselines3.common.torch_layers import create_mlp
from torch import nn


class PerceiverNet(nn.Module):
    """
    Inspired by Necto's EARL model and Perceiver https://arxiv.org/abs/2103.03206
    """

    class PerceiverBlock(nn.Module):
        def __init__(self, d_model, ca_nhead):
            super().__init__()
            self.cross_attention = nn.MultiheadAttention(d_model, ca_nhead, batch_first=True)
            self.linear1 = nn.Linear(d_model, d_model)
            self.linear2 = nn.Linear(d_model, d_model)
            self.activation = nn.ReLU()
            # No batch norm
            # No dropout, introducing more variance in RL is bad
            self._reset_parameters()

        def _reset_parameters(self):
            r"""
            *Following PyTorch Transformer implementation.

            Initiate parameters in the transformer model.
            """
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        def forward(self, latent, byte, key_padding_mask=None):
            # attention outputs only
            out = self.cross_attention(latent, byte, byte, key_padding_mask)[0] + latent  # skip connection
            return self.linear2(self.activation(self.linear1(out))) + out  # skip connection

    def __init__(self,
                 query_dims,
                 kv_dims,
                 hidden_dims=256,
                 n_layers=4,
                 ca_nhead=4,
                 n_preprocess_layers=1,
                 n_postprocess_layers=0,
                 recurrent=False):
        super(PerceiverNet, self).__init__()

        self.query_dims = query_dims
        self.kv_dims = kv_dims
        self.hidden_dims = hidden_dims
        self.n_layers = n_layers
        self.ca_nhead = ca_nhead
        self.n_preprocess_layers = n_preprocess_layers
        self.n_postprocess_layers = n_postprocess_layers
        self.recurrent = recurrent

        self.query_preprocess = nn.Sequential(*create_mlp(query_dims, -1, [hidden_dims] * n_preprocess_layers))
        self.kv_preprocess = nn.Sequential(*create_mlp(kv_dims, -1, [hidden_dims] * n_preprocess_layers))
        # If the network is recurrent repeat the same block at each layer
        if recurrent:
            self.perceiver_blocks = nn.ModuleList([self.PerceiverBlock(hidden_dims, ca_nhead)] * n_layers)
        # Otherwise, create new blocks for each layer
        else:
            self.perceiver_blocks = nn.ModuleList([self.PerceiverBlock(hidden_dims, ca_nhead) for _ in range(n_layers)])

        if self.n_postprocess_layers > 0:
            self.postprocess = nn.Sequential(*create_mlp(hidden_dims,
                                                         hidden_dims,
                                                         [hidden_dims] * (n_preprocess_layers - 1)))
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

        self.actor = PerceiverNet(**net_arch[0])
        self.critic = PerceiverNet(**net_arch[1])

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

    def forward_actor(self, features):
        key_padding_mask = features[:, 1:, -1]
        query = features[:, [0], :-1]
        obs = features[:, 1:, :-9]
        return self.actor(query, obs, key_padding_mask).squeeze(1)

    def forward_critic(self, features):
        key_padding_mask = features[:, 1:, -1]
        query = features[:, [0], :-1]
        obs = features[:, 1:, :-9]
        return self.critic(query, obs, key_padding_mask).squeeze(1)
