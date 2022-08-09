from typing import Union

import torch as th
from stable_baselines3.common.utils import get_device
from torch import nn

from .perceiver import PerceiverNet
from .necto_perceiver import NectoPerceiverNet


class SeqRewardPredictionNetwork(nn.Module):
    def __init__(self,
                 actor: Union[PerceiverNet, NectoPerceiverNet],
                 seq_len: int,
                 n_heads: int = 4,
                 device: Union[th.device, str] = "auto",
                 ):
        super(SeqRewardPredictionNetwork, self).__init__()
        device = get_device(device)

        self.seq_len = seq_len

        self.actor = actor
        n_latent = actor.latent_dims

        self.cross_attn = nn.MultiheadAttention(n_latent, n_heads, batch_first=True, device=device)
        self.norm = nn.LayerNorm(n_latent, device=device)

        self.lstm_hidden_size = 32
        self.lstm = nn.LSTM(input_size=n_latent,
                            hidden_size=self.lstm_hidden_size,
                            num_layers=1,
                            batch_first=True,
                            device=device)
        self.linear = nn.Linear(self.lstm_hidden_size, 3, device=device)

    def forward(self,
                batch_size: int,
                query: th.Tensor,
                obs: th.Tensor,
                key_padding_mask: Union[None, th.Tensor] = None) -> th.Tensor:
        """
        :param batch_size: Batch dimension used to reshape the cross-attention output
        :param query: Query tensor of shape (batch_size, seq_len, 1, query_dims)
        :param obs: Key/value tensor of shape (batch_size, seq_len, n_kv, kv_dims)
        :param key_padding_mask: Key padding boolean mask of shape (batch_size, seq_len, n_kv)
        :return: Tensor of shape (batch_size, 3)
        """
        q_emb = self.actor.query_norm(self.actor.query_preprocess(query))
        kv_emb = self.actor.kv_norm(self.actor.kv_preprocess(obs))

        # squeeze player dimension
        out = self.norm(self.cross_attn(q_emb, kv_emb, kv_emb, key_padding_mask)[0] + q_emb).squeeze(1)

        # batch * sequence, n_latent -> batch, sequence, n_latent
        out = out.view((batch_size, self.seq_len, out.shape[1]))

        _, (h_n, _) = self.lstm.forward(out)
        out = h_n.view(-1, self.lstm_hidden_size)

        return self.linear(out)


class DeepStateRepresentationNetwork(nn.Module):
    def __init__(self,
                 actor: Union[PerceiverNet, NectoPerceiverNet],
                 obs_shape: int,
                 n_heads: int = 4,
                 device: Union[th.device, str] = "auto",
                 ):
        super().__init__()
        device = get_device(device)

        self.actor = actor
        n_latent = actor.latent_dims

        self.cross_attn = nn.MultiheadAttention(n_latent, n_heads, batch_first=True, device=device)
        self.norm = nn.LayerNorm(n_latent, device=device)

        self.encoder = nn.Sequential(
            nn.Linear(n_latent, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU()
        ).to(device)

        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, obs_shape),
            nn.ReLU()
        ).to(device)

    def forward(self, query: th.Tensor, obs: th.Tensor, key_padding_mask: Union[None, th.Tensor] = None) -> th.Tensor:
        """
        :param query: Query tensor of shape (batch_size, 1, query_dims)
        :param obs: Key/value tensor of shape (batch_size, n_kv, kv_dims)
        :param key_padding_mask: Key padding boolean mask of shape (batch_size, n_kv)
        :return: Tensor of shape (batch_size, obs_shape)
        """
        q_emb = self.actor.query_norm(self.actor.query_preprocess(query))
        kv_emb = self.actor.kv_norm(self.actor.kv_preprocess(obs))

        # squeeze player dimension
        out = self.norm(self.cross_attn(q_emb, kv_emb, kv_emb, key_padding_mask)[0] + q_emb).squeeze(1)

        out = self.encoder(out)
        return self.decoder(out)
