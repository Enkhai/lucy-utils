from stable_baselines3.common.torch_layers import create_mlp
from torch import nn


class MLPPolicyNetwork(nn.Module):
    def __init__(self, n_features, hidden_dims, n_layers, activation_fn=nn.ReLU, dropout=0.1, output_features=None):
        super(MLPPolicyNetwork, self).__init__()
        m = (nn.Linear(n_features, hidden_dims),
             activation_fn(),
             nn.Dropout(dropout))
        m += (nn.Linear(hidden_dims, hidden_dims),
              activation_fn(),
              nn.Dropout(dropout),) * (n_layers - 2)
        if output_features:
            m += (nn.Linear(hidden_dims, output_features),)
        else:
            m += (nn.Linear(hidden_dims, hidden_dims), activation_fn(), nn.Dropout(dropout))
        self.model = nn.Sequential(*m)

        self.latent_dim_pi, self.latent_dim_vf = (hidden_dims,) * 2

    def forward(self, features):
        return self.model(features), self.model(features)

    def forward_actor(self, features):
        return self.model(features)

    def forward_critic(self, features):
        return self.model(features)


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

    def __init__(self, net_arch):
        super(PerceiverNet, self).__init__()

        # Parse arguments
        self.query_dims = net_arch["query_dims"]
        self.kv_dims = net_arch["kv_dims"]
        self.hidden_dims = net_arch["hidden_dims"] if "hidden_dims" in net_arch else 256
        self.n_layers = net_arch["n_layers"] if "n_layers" in net_arch else 4
        self.ca_nhead = net_arch["n_ca_heads"] if "n_ca_heads" in net_arch else 4
        if "n_preprocess_layers" in net_arch and net_arch["n_preprocess_layers"] > 1:
            self.n_preprocess_layers = net_arch["n_preprocess_layers"]
        else:
            self.n_preprocess_layers = net_arch["n_preprocess_layers"] = 1

        if "n_postprocess_layers" in net_arch and net_arch["n_postprocess_layers"] > 0:
            self.n_postprocess_layers = net_arch["n_postprocess_layers"]
        else:
            self.n_postprocess_layers = net_arch["n_postprocess_layers"] = 0

        # Build the architecture
        self.query_preprocess = nn.Sequential(*create_mlp(self.query_dims,
                                                          -1,
                                                          [self.hidden_dims] * self.n_preprocess_layers))
        self.kv_preprocess = nn.Sequential(*create_mlp(self.kv_dims,
                                                       -1,
                                                       [self.hidden_dims] * self.n_preprocess_layers))
        # If the network is recurrent repeat the same block at each layer
        if "recurrent" in net_arch and net_arch["recurrent"]:
            self.perceiver_blocks = nn.ModuleList([self.PerceiverBlock(self.hidden_dims,
                                                                       self.ca_nhead)] * self.n_layers)
        # Otherwise, create new blocks for each layer
        else:
            self.perceiver_blocks = nn.ModuleList([self.PerceiverBlock(self.hidden_dims,
                                                                       self.ca_nhead)
                                                   for _ in range(self.n_layers)])

        if self.n_postprocess_layers > 0:
            self.postprocess = nn.Sequential(*create_mlp(self.hidden_dims,
                                                         self.hidden_dims,
                                                         [self.hidden_dims] * (self.n_preprocess_layers - 1)))
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

        self.actor = PerceiverNet(net_arch[0])
        self.critic = PerceiverNet(net_arch[1])

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
