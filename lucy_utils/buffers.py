from typing import NamedTuple, Optional, Generator

import numpy as np
import torch as th
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env import VecNormalize


class AuxRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    rewards: th.Tensor
    episode_starts: th.Tensor
    obs_sequences: th.Tensor


class AuxRolloutBuffer(RolloutBuffer):
    def __init__(self,
                 make_obs_sequences: bool = False,
                 sequence_len: int = 1,
                 *args,
                 **kwargs):
        self.make_obs_sequences = make_obs_sequences
        self.sequence_len = sequence_len
        super(AuxRolloutBuffer, self).__init__(*args, **kwargs)

    def create_seq_batch(self):
        if not self.make_obs_sequences:
            self.obs_sequences = th.tensor([])
            return

        observations, episode_starts = th.tensor(self.observations), th.tensor(self.episode_starts).reshape(-1)
        rollout_size = self.n_envs * self.buffer_size
        # * assume swap_and_flatten
        env_indices = th.arange(rollout_size).view(self.n_envs, self.buffer_size).long()

        # preliminaries
        pad_len = self.sequence_len - 1
        obs_pad = th.zeros((pad_len,) + self.obs_shape)

        # to-be-mapping
        xs = []
        for env_ids in env_indices:
            # env episode starts and observations
            ep_starts = episode_starts[env_ids]
            obss = observations[env_ids]

            # pad the observations at
            #  - the beginning: to fill zero observations for the first observation sequences
            #  - the end: to build proper replacing sequences for sequences containing episode starts
            padded_obss = th.cat((obs_pad, obss, obs_pad))

            # padded obss indices
            ep_starts_ids = ep_starts.nonzero()
            ep_starts_ids = th.cat([ep_starts_ids + i for i in range(2 * pad_len)], -1)
            # does a sequence need replacing?
            has_seq_ep_start = ep_starts_ids[:, :pad_len]
            has_seq_ep_start = has_seq_ep_start[has_seq_ep_start < self.buffer_size]

            # build replacing sequences
            padded_obss2 = padded_obss[ep_starts_ids]
            padded_obss2[:, :pad_len] = obs_pad
            padded_seq_obss2 = th.flatten(padded_obss2.unfold(1, self.sequence_len, 1),
                                          0, 1).movedim(-1, 1)[:has_seq_ep_start.shape[0]].contiguous()

            # make regular observation sequences for the env
            padded_seq_obss = padded_obss[:self.buffer_size + pad_len].unfold(0,
                                                                              self.sequence_len,
                                                                              1).movedim(-1, 1).contiguous()

            # replace accordingly for sequences that contain episode starts
            padded_seq_obss[has_seq_ep_start] = padded_seq_obss2

            xs.append(padded_seq_obss)

        self.obs_sequences = th.cat(xs)

    def get(self, batch_size: Optional[int] = None) -> Generator[AuxRolloutBufferSamples, None, None]:
        assert self.full, ""
        rollout_size = self.buffer_size * self.n_envs
        indices = np.random.permutation(rollout_size)
        # Prepare the data
        if not self.generator_ready:

            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "rewards",
                "episode_starts"
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.create_seq_batch()
            self.generator_ready = True
        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = rollout_size

        start_idx = 0
        while start_idx < rollout_size:
            yield self._get_samples(indices[start_idx: start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> AuxRolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.rewards[batch_inds].flatten(),
            self.episode_starts[batch_inds].flatten(),
            self.obs_sequences[batch_inds] if self.make_obs_sequences else self.obs_sequences,
        )
        return AuxRolloutBufferSamples(*tuple(map(self.to_torch, data)))
