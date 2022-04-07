from typing import Any

import numpy as np
from rlgym.utils import ObsBuilder, common_values
from rlgym.utils.gamestates import GameState, PlayerData


class AttentionObs(ObsBuilder):
    """
    Observation builder suitable for attention models.\n
    Inspired by Necto's obs builder. Missing demo timers, boost timers and boost pad locations.

    Returns an observation tensor of shape (1 `q` player + 1 ball + 6 players, 32 features).

    Features:
     - 0-4 flags: main player, teammate, opponent, ball
     - 4-7: (relative) normalized position
     - 7-10: (relative) normalized linear velocity
     - 10-13: normalized angular velocity
     - 13-16: forward vector
     - 16-19: upward vector
     - 19: boost amount
     - 20: on ground flag
     - 21: has flip flag
     - 22: demo flag
     - 23-31: previous action
     - 31: key padding mask boolean

    The key padding mask is useful in maintaining multiple matches of different sizes and allowing the model
    to play in a variety of settings simultaneously
    """
    # Boost pad locations may also be useful for a model trained to pick up and maintain boost
    # When building a hierarchical architecture, different parts of the observation can be used by different submodels,
    # depending on the purpose

    current_state = None
    current_obs = None

    # Inversion vector - needed for invariance, used to invert the x and y axes for the orange team observation
    _invert = np.array([1] * 4 + [-1, -1, 1] * 5 + [1] * 4 + [1] * 8 + [1])

    def __init__(self, n_players=6):
        super(AttentionObs, self).__init__()
        self.n_players = n_players

    def reset(self, initial_state: GameState):
        pass

    def _update_state_and_obs(self, state: GameState):
        obs = np.zeros((1 + self.n_players, 23 + 8 + 1))
        obs[:, -1] = 1  # key padding mask

        # Ball
        ball = state.ball
        obs[0, 3] = 1  # ball flag
        # Ball and car position and velocity may use the same scale since they are treated similarly as objects
        # in the observation
        obs[0, 4:7] = ball.position / common_values.CAR_MAX_SPEED
        obs[0, 7:10] = ball.linear_velocity / common_values.CAR_MAX_SPEED
        obs[0, 10:13] = ball.angular_velocity / common_values.CAR_MAX_ANG_VEL
        # no forward, upward, boost amount, touching ground, flip and demoed info for ball
        obs[0, -1] = 0  # mark non-padded

        # Players
        for i, p in zip(range(1, len(state.players) + 1), state.players):
            if p.team_num == common_values.BLUE_TEAM:  # team flags
                obs[i, 1] = 1
            else:
                obs[i, 2] = 1
            p_car = p.car_data
            obs[i, 4:7] = p_car.position / common_values.CAR_MAX_SPEED
            obs[i, 7:10] = p_car.linear_velocity / common_values.CAR_MAX_SPEED
            obs[i, 10:13] = p_car.angular_velocity / common_values.CAR_MAX_ANG_VEL
            obs[i, 13:16] = p_car.forward()
            obs[i, 16:19] = p_car.up()
            # we could also use p_car.right(), steering might be useful
            obs[i, 19] = p.boost_amount
            obs[i, 20] = p.on_ground
            obs[i, 21] = p.has_flip
            obs[i, 22] = p.is_demoed
            obs[i, -1] = 0  # mark non-padded

        self.current_obs = obs
        self.current_state = state

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:
        # No need to update the state until model produces output for all cars
        # When it does and the state changes, update
        if state != self.current_state:
            self._update_state_and_obs(state)

        obs = self.current_obs.copy()

        player_idx = state.players.index(player) + 1  # plus one because ball is first
        obs[player_idx, 0] = 1  # player flag
        if player.team_num == common_values.ORANGE_TEAM:  # if orange team
            obs[:, [1, 2]] = obs[:, [2, 1]]  # swap team flags
            obs *= self._invert  # invert x and y axes

        query = obs[[player_idx], :]
        query[0, -9:-1] = previous_action  # add previous action to player query

        obs[:, 4:10] -= query[0, 4:10]  # relative position and linear velocity
        obs = np.concatenate([query, obs])  # should we remove the player from the observation?

        # Dictionary spaces are not supported with multi-instance envs,
        # so we need to put the outputs (query, obs and mask) into a single numpy array
        return obs
