from collections import deque
from typing import Any

import numpy as np
from rlgym.utils import ObsBuilder, common_values
from rlgym.utils.gamestates import GameState, PlayerData


class GraphAttentionObs(ObsBuilder):
    """
    Observation builder suitable for attention models with a previous action stacker and adjacency vectors.

    Inspired by Necto's obs builder. Demo and boost flags are used in place of timers.

    Returns an observation matrix of shape (1 player query + (1 ball + `n_players` + `_n_boost_pads` key/value objects),
    33+ or 40+ features).

    Features:
     - 0-5 flags: main player, teammate, opponent, ball, boost
     - 5-8: (relative) normalized position
     - 8-11: (relative) normalized linear velocity
     - 11-14: normalized angular velocity
     - 14-17: forward vector
     - 17-20: upward vector
     - 20: boost amount
     - 21: on ground flag
     - 22: has flip flag
     - 23: demo / boost flag
    If the observation is a regular observation:
     - 24-(24 + 8 * stack size): previous actions
     - -1: key padding mask boolean
    If the observation is a graph observation:
     - 24-(24 + 8 * stack size): previous actions
     - (-1 + `n_players` + `_n_boost_pads`)-(-1): distance adjacency vector
     - -1: key padding mask boolean

    Adjacency vectors are computed among observation objects using a `Player to ball distance` reward logic.
    Dispersion and density factors affect distance values. Additionally, adjacency vectors are shifted in order
    to have a mean of 1. Adjacency vectors are always positive. Those characteristics make them useful for encoding
    spatial information in the form of graph edge weights and can be used for weighting key/value features.

    The key padding mask is useful in maintaining multiple matches of different sizes and allowing attention-based
    models to play in a variety of settings simultaneously.
    """
    _boost_locations = np.array(common_values.BOOST_LOCATIONS)

    current_state = None
    current_obs = None
    default_action = np.zeros(8)

    def __init__(self, n_players=6, stack_size=1, add_boost_pads=False, graph_obs=False, dispersion=1, density=1):
        """
        :param n_players: Maximum number of players in the observation
        :param stack_size: Number of previous actions to stack
        :param add_boost_pads: Dictates whether boost pads should be included in the observation
        :param graph_obs: Dictates whether adjacency vectors are computed and returned in the observation
        :param dispersion: Dispersion factor for object-to-object distance values
        :param density: Density factor for object-to-object distance values
        """
        super(GraphAttentionObs, self).__init__()
        assert dispersion > 0 and density > 0, "You must specify dispersion and density values larger than 0"
        assert stack_size >= 0, "stack_size must be larger than or equal to 0"

        self.n_players = n_players
        _n_boost_pads = add_boost_pads * 34  # 0 or 34
        self._invert = np.array([1] * 5 +  # flags
                                [-1, -1, 1] * 5 +  # position, lin vel, ang vel, forward, up
                                [1] * 4 +  # flags
                                [1] * (8 * stack_size) +  # previous actions
                                [1] * ((1 + n_players + _n_boost_pads) * graph_obs) +  # adjacency vector
                                [1])  # key padding mask
        self._norm = np.ones_like(self._invert, dtype=float)
        self._norm[5:11] = common_values.CAR_MAX_SPEED
        self._norm[11:14] = common_values.CAR_MAX_ANG_VEL

        self._obs_shape = (1 + n_players + _n_boost_pads, self._invert.shape[0])  # used for updating current_obs
        self._action_offset = 1 + (1 + n_players + _n_boost_pads) * graph_obs  # adjacency matrix and action stacking

        self.obs_shape = (1 + 1 + n_players + _n_boost_pads, self._invert.shape[0])  # actual observation shape

        self.add_boost_pads = add_boost_pads

        self.dispersion = dispersion
        self.density = density
        self.graph_obs = graph_obs

        self.stack_size = stack_size
        self.action_stack = [deque(maxlen=stack_size) for _ in range(64)]
        for i in range(64):
            self.blank_stack(i)

    def blank_stack(self, index: int) -> None:
        for _ in range(self.stack_size):
            self.action_stack[index].appendleft(self.default_action)

    def reset(self, initial_state: GameState):
        for p in initial_state.players:
            self.blank_stack(p.car_id)

    def _update_state_and_obs(self, state: GameState):
        obs = np.zeros(self._obs_shape)
        obs[:, -1] = 1  # key padding mask

        # Ball
        ball = state.ball
        obs[0, 3] = 1  # ball flag
        obs[0, 5:8] = ball.position
        obs[0, 8:11] = ball.linear_velocity
        obs[0, 11:14] = ball.angular_velocity
        # no forward, upward, boost amount, touching ground, flip and demoed info for ball
        obs[0, -1] = 0  # mark non-padded

        # Players
        for i, p in zip(range(1, len(state.players) + 1), state.players):
            if p.team_num == common_values.BLUE_TEAM:  # team flags
                obs[i, 1] = 1
            else:
                obs[i, 2] = 1
            p_car = p.car_data
            obs[i, 5:8] = p_car.position
            obs[i, 8:11] = p_car.linear_velocity
            obs[i, 11:14] = p_car.angular_velocity
            obs[i, 14:17] = p_car.forward()
            obs[i, 17:20] = p_car.up()
            obs[i, 20] = p.boost_amount
            obs[i, 21] = p.on_ground
            obs[i, 22] = p.has_flip
            obs[i, 23] = p.is_demoed
            obs[i, -1] = 0  # mark non-padded

        # Boost pads
        if self.add_boost_pads:
            n = 1 + self.n_players
            obs[n:, 4] = 1  # boost flag
            obs[n:, 5:8] = self._boost_locations
            obs[n: 20] = 0.12 + 0.88 * (self._boost_locations[:, 2] > 72)
            # no velocities, rotation, touching ground, flip and demoed info for boost pads
            obs[n:, 23] = state.boost_pads
            obs[n:, -1] = 0  # mark non-padded

        obs /= self._norm

        if self.graph_obs:
            self._compute_adjacency_matrix(obs)

        self.current_obs = obs
        self.current_state = state

    def _compute_adjacency_matrix(self, obs):
        # TODO: Implement support for self-connection weights of 1
        positions = obs[:, 5:8]
        distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
        # player to ball distance reward logic
        distances = np.exp(-0.5 * distances / self.dispersion) ** (1 / self.density)

        mask = obs[:, -1].astype(bool)
        mask2 = np.repeat(mask[None, :], mask.shape[0], 0)
        mask2 += mask2.T

        distances[mask2] = 0  # zero out padded objects

        masked_distances = distances[~mask][:, ~mask]
        masked_distances /= masked_distances.mean(1)[:, None]  # ensuring the mean is always 1
        distances[~mask2] = masked_distances.flatten()

        obs[:, -self._action_offset: -1] = distances

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:
        self.action_stack[player.car_id].appendleft(previous_action)

        if state != self.current_state:
            self._update_state_and_obs(state)

        obs = self.current_obs.copy()

        player_idx = state.players.index(player) + 1  # ball is first
        obs[player_idx, 0] = 1  # main player flag
        # if orange swap teams and invert x and y
        if player.team_num == common_values.ORANGE_TEAM:
            obs[:, [1, 2]] = obs[:, [2, 1]]
            obs *= self._invert

        query = obs[[player_idx], :]
        # add previous actions to player query
        query[0, -(self._action_offset + 8 * self.stack_size):-self._action_offset] = np.concatenate(
            self.action_stack[player.car_id] or [np.array([])])

        obs[:, 5:11] -= query[0, 5:11]  # relative position and linear velocity
        obs = np.concatenate([query, obs])

        # Dictionary spaces are not supported with multi-instance envs,
        # so we need to put the outputs (query, obs and mask) into a single numpy array
        return obs
