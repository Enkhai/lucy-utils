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
