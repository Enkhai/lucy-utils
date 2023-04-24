from typing import Union, List

import torch as th
from stable_baselines3.common.torch_layers import create_mlp
from torch import nn

from lucy_utils.actors import NextoActor


class PerceiverNet(nn.Module):
    """
    Perceiver-like network architecture inspired by Necto's EARL model and Perceiver https://arxiv.org/abs/2103.03206.

    Resembles the Transformer encoder architecture more closely.
    """

    class PerceiverBlock(nn.Module):
        def __init__(self,
                     d_model: int,
                     ca_nhead: int,
                     d_model_feedforward_mult: int = 4,
                     dim_feedforward: Union[None, int] = None,
                     use_norm: bool = True):
            """
            :param d_model: The number of expected features in the input
            :param ca_nhead: The number of heads in the cross-attention layer
            :param d_model_feedforward_mult: The multiplier for the number of features in the feedforward
             network model. Applies only when dim_feedforward is None.
            :param dim_feedforward: The number of features of the feedforward network model
            :param use_norm: Dictates whether to normalize the latent and byte inputs and the cross-attention output
            """
            super().__init__()
            if dim_feedforward is None:
                dim_feedforward = d_model_feedforward_mult * d_model

            self.cross_attention = nn.MultiheadAttention(d_model, ca_nhead, batch_first=True)
            self.linear1 = nn.Linear(d_model, dim_feedforward)
            self.linear2 = nn.Linear(dim_feedforward, d_model)
            self.activation = nn.ReLU()

            if use_norm:
                self.norm1 = nn.LayerNorm(d_model)
                self.norm2 = nn.LayerNorm(d_model)
            else:
                self.norm1 = nn.Identity()
                self.norm2 = nn.Identity()

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
            out = self.cross_attention(latent, byte, byte, key_padding_mask)[0] + latent
            out = self.norm1(out)
            out = self.linear2(self.activation(self.linear1(out))) + out
            return self.norm2(out)

    def __init__(self,
                 # Preprocessing
                 query_dims: int,
                 kv_dims: int,
                 hidden_dims: int = 128,  # + attention & postprocessing
                 n_preprocess_layers: int = 2,
                 # Attention layers
                 ca_nhead: int = 4,
                 n_layers: int = 2,
                 feedforward_dim_mult: int = 4,
                 feedforward_dims: Union[None, int] = None,
                 use_norm: bool = True,
                 recurrent: bool = False,
                 # Postprocessing
                 n_postprocess_layers: int = 0,
                 # Action predictor
                 player_emb_net_shape: List[int] = None,
                 action_emb_net_shape: List[int] = None,
                 ):
        """
        :param query_dims: Number of expected features for the query
        :param kv_dims: Number of expected features for the key and value
        :param hidden_dims: Number of dimensions for the pre- and postprocessing layers
        :param n_preprocess_layers: Number of preprocessing layers
        :param ca_nhead: Number of cross-attention heads in each Perceiver block
        :param n_layers: Number of Perceiver blocks
        :param feedforward_dim_mult: Multiplier used for the number of dimensions of Perceiver feedforward layers.
        :param feedforward_dims: Number of dimensions of the feedforward layers in Perceiver blocks.
         If `None`, the number of dimensions of feedforward layers become `feedforward_dim_mult * hidden_dims`.
        :param use_norm: Dictates whether to normalize the latent and byte inputs and the cross-attention output
         in Perceiver blocks, as well as the final block output
        :param recurrent: Dictates whether Perceiver blocks are recurrent, i.e. the same module used for all blocks
        :param n_postprocess_layers: Number of postprocessing layers
        :param player_emb_net_shape: *Nexto actor only*. MLP network shape for processing latent network output.
         Produces embeddings for players. A dot product operation is performed on action and player embeddings to
         produce action logits.
         If either player_emb_net_shape or actor_emb_net_shape is `None`, latent output is not processed.
         Requires setting `is_nexto` in `ActorCriticAttnPolicy`.
        :param action_emb_net_shape: *Nexto actor only*. MPL network shape for processing possible Nexto actions.
         Produces embeddings for actions. A dot product operation is performed on action and player embeddings to
         produce action logits.
         If either actor_emb_net_shape or player_emb_net_shape is `None`, latent output is not processed.
         Requires setting `is_nexto` in `ActorCriticAttnPolicy`.
        """
        super(PerceiverNet, self).__init__()

        assert n_preprocess_layers > 1, "You must specify at least one preprocessing layer"

        self.latent_dims = hidden_dims  # required for SB3 policy

        self.query_preprocess = nn.Sequential(*create_mlp(query_dims, -1, [hidden_dims] * n_preprocess_layers))
        self.kv_preprocess = nn.Sequential(*create_mlp(kv_dims, -1, [hidden_dims] * n_preprocess_layers))

        self.query_norm, self.kv_norm = nn.LayerNorm(hidden_dims), nn.LayerNorm(hidden_dims)

        if recurrent:
            self.perceiver_blocks = nn.ModuleList([self.PerceiverBlock(hidden_dims,
                                                                       ca_nhead,
                                                                       feedforward_dim_mult,
                                                                       feedforward_dims,
                                                                       use_norm)] * n_layers)
        else:
            self.perceiver_blocks = nn.ModuleList([self.PerceiverBlock(hidden_dims,
                                                                       ca_nhead,
                                                                       feedforward_dim_mult,
                                                                       feedforward_dims,
                                                                       use_norm) for _ in range(n_layers)])

        if n_postprocess_layers > 0:
            self.postprocess = nn.Sequential(*create_mlp(hidden_dims, -1, [hidden_dims] * n_postprocess_layers))
        else:
            self.postprocess = nn.Identity()

        if player_emb_net_shape and action_emb_net_shape:
            self.player_emb_net = nn.Sequential(*create_mlp(hidden_dims,
                                                            -1,
                                                            player_emb_net_shape))
            self.action_emb_net = nn.Sequential(*create_mlp(hidden_dims,
                                                            -1,
                                                            action_emb_net_shape))
            self._actions = th.from_numpy(NextoActor.make_lookup_table()).float()

    def forward(self, query, obs, key_padding_mask=None):
        q_emb = self.query_norm(self.query_preprocess(query))
        kv_emb = self.kv_norm(self.kv_preprocess(obs))

        for block in self.perceiver_blocks:
            q_emb = block(q_emb, kv_emb, key_padding_mask)  # update latent only

        out = self.postprocess(q_emb)

        if '_actions' in self.__dict__:
            player_emb = self.player_emb_net(out)
            act_emb = self.action_emb_net(self._actions.to(player_emb.device))

            out = th.einsum('bad,bpd->bpa', act_emb, player_emb)

        return out
