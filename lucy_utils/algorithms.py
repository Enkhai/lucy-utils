import time
from abc import ABC
from typing import Optional, Union, Type, Dict, Any

import numpy as np
import torch as th
import torch.nn.functional as F
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import safe_mean, get_schedule_fn, explained_variance

from .buffers import AuxRolloutBuffer
from .policies import AuxACAttnPolicy


class DeviceAlternatingOnPolicyAlgorithm(OnPolicyAlgorithm, ABC):
    """
    An on-policy algorithm abstract class that subclasses OnPolicyAlgorithm and alternates between collecting rollouts
    on the CPU and training the policy on the GPU by default.\n
    Set `dva=False` during initialization to disable the device-alternating property.\n
    Other arguments and keyworded arguments remain the same.
    """

    def __init__(self, *args, dva=True, **kwargs):
        super(DeviceAlternatingOnPolicyAlgorithm, self).__init__(*args, **kwargs)
        self.dva = dva

    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 1,
            eval_env: Optional[GymEnv] = None,
            eval_freq: int = -1,
            n_eval_episodes: int = 5,
            tb_log_name: str = "OnPolicyAlgorithm",
            eval_log_path: Optional[str] = None,
            reset_num_timesteps: bool = True,
    ) -> "OnPolicyAlgorithm":
        # Lines marked with # ++ are lines that have been added

        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps,
            tb_log_name
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:

            if self.dva:  # ++
                self.device = "cpu"  # ++
                self.policy.cpu()  # ++
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer,
                                                      n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                fps = int((self.num_timesteps - self._num_timesteps_at_start) / (time.time() - self.start_time))
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record("rollout/ep_rew_mean",
                                       safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean",
                                       safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

            if self.dva:  # ++
                self.device = "cuda"  # ++
                self.policy.cuda()  # ++
            self.train()

        callback.on_training_end()

        return self


class DeviceAlternatingPPO(PPO, DeviceAlternatingOnPolicyAlgorithm):
    """
    A PPO algorithm variant that alternates between collecting rollouts
    on the CPU and training the policy on the GPU by default.\n
    Set `dva=False` during initialization to disable the device-alternating property.\n
    Other arguments and keyworded arguments remain the same.
    """
    pass


class AuxPPO(DeviceAlternatingPPO):
    """
    A device-alternating PPO variant for incorporating auxiliary tasks into the training pipeline.
    Policy arguments should also include:
     - 'use_rp': boolean for reward prediction learning.
     - 'rp_arch': string for reward prediction architecture. Can be either "base", "deep" or "deep2".
     - 'use_sr': boolean for state reconstruction learning.
     - 'sr_arch`: string for state representation architecture. Can be either "base", "deep" or "seq".
     - 'rp_seq_len': integer for controlling the length of the reward prediction sequence. Works with "seq" 'sr_arch'.
     - 'zero_rew_threshold`: float for controlling which rewards are considered to be in the `zero` class
    """

    def __init__(self,
                 policy: Type[AuxACAttnPolicy],
                 env: Union[GymEnv, str],
                 learning_rate: Union[float, Schedule] = 3e-4,
                 n_steps: int = 2048,
                 batch_size: int = 64,
                 n_epochs: int = 10,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_range: Union[float, Schedule] = 0.2,
                 clip_range_vf: Union[None, float, Schedule] = None,
                 normalize_advantage: bool = True,
                 ent_coef: float = 0.0,
                 vf_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 use_sde: bool = False,
                 sde_sample_freq: int = -1,
                 target_kl: Optional[float] = None,
                 tensorboard_log: Optional[str] = None,
                 create_eval_env: bool = False,
                 policy_kwargs: Optional[Dict[str, Any]] = None,
                 verbose: int = 0,
                 seed: Optional[int] = None,
                 device: Union[th.device, str] = "auto",
                 _init_setup_model: bool = True,
                 ):
        # Aux args init
        self.use_rp = policy_kwargs['use_rp']
        self.use_sr = policy_kwargs['use_sr']
        self.rp_seq_len = policy_kwargs['rp_seq_len']
        self.zero_rew_threshold = policy_kwargs['zero_rew_threshold']

        # delete non-necessary policy kwarg
        del policy_kwargs['zero_rew_threshold']

        super(AuxPPO, self).__init__(policy, env, learning_rate, n_steps, batch_size, n_epochs, gamma, gae_lambda,
                                     clip_range, clip_range_vf, normalize_advantage, ent_coef, vf_coef, max_grad_norm,
                                     use_sde, sde_sample_freq, target_kl, tensorboard_log, create_eval_env,
                                     policy_kwargs, verbose, seed, device, _init_setup_model)

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        self.rollout_buffer = AuxRolloutBuffer(
            self.rp_seq_len,
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        self.policy = self.policy_class(  # pytype:disable=not-instantiable
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " \
                                               "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def one_hot_targets(self, rewards):
        # separate reward types
        positive_rew_indices = (rewards > self.zero_rew_threshold).nonzero()
        negative_rew_indices = (rewards < -self.zero_rew_threshold).nonzero()
        zero_rew_indices = (rewards.abs() <= self.zero_rew_threshold).nonzero()
        y = th.zeros((rewards.shape[0], 3), device=self.device)
        y[positive_rew_indices, 0] = 1  # positives
        y[negative_rew_indices, 1] = 1  # negatives
        y[zero_rew_indices, 2] = 1  # zeros
        return y

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        # AUX losses logging
        rp_losses = []
        sr_losses = []

        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []

            # Do a complete pass on the rollout buffer
            batch_counter = -1
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                batch_counter += 1
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Auxiliary losses
                if self.use_sr:
                    sr_obs_pred = self.policy.forward_sr(rollout_data.observations)
                    _, sr_obs, _ = self.policy.mlp_extractor.extract_features(rollout_data.observations)
                    sr_loss = F.smooth_l1_loss(sr_obs_pred, sr_obs.view(self.batch_size, -1))
                    sr_losses.append(sr_loss.item())
                    loss += sr_loss
                if self.use_rp:
                    current_indices = self.rollout_buffer.current_indices
                    rp_x = self.rollout_buffer.mapping[current_indices].to(self.device)
                    rp_y = self.one_hot_targets(rollout_data.rewards)
                    rp_y_pred = self.policy.forward_rp(rp_x)
                    rp_loss = F.cross_entropy(rp_y_pred, rp_y)
                    rp_losses.append(rp_loss.item())
                    loss += rp_loss
                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break
                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            if not continue_training:
                break

        # Free memory
        del rp_x, rp_y, self.rollout_buffer.mapping
        th.cuda.empty_cache()

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
        # Aux log
        if self.use_rp:
            self.logger.record("train/rp_loss", np.mean(rp_losses))
        if self.use_sr:
            self.logger.record("train/sr_loss", np.mean(sr_losses))
