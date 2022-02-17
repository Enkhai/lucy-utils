from typing import Any

import numpy as np
from rlgym.utils import ObsBuilder, common_values
from rlgym.utils.gamestates import GameState, PlayerData


class SimpleObs(ObsBuilder):
    """
    Simple observation builder for a ball and one car only
    Observation space is of shape 1 * 2 * 20:
    1 (batch)
    * 2 (1 ball + 1 car)
    * 20 (2 (car and ball flags)
        + 9 ((relative) standardized position, linear velocity and angular velocity  3-d vectors)
        + 6 (forward and upward rotation axes 3-d vectors)
        + 3 (boost, touching ground and has flip flags))

    If flatten is true, it simply returns a vector of length 40 (2 * 20)
    """
    POS_STD = 3000

    def __init__(self, flatten: bool = False):
        super(SimpleObs, self).__init__()
        self.flatten = flatten

    def reset(self, initial_state: GameState):
        pass

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:
        ball = state.ball
        car = player.car_data

        ball_obs = np.concatenate([[1, 0],
                                   (ball.position - car.position) / self.POS_STD,
                                   (ball.linear_velocity - car.linear_velocity) / common_values.BALL_MAX_SPEED,
                                   ball.angular_velocity / common_values.CAR_MAX_ANG_VEL,
                                   np.zeros(6),
                                   np.zeros(3)])
        car_obs = np.concatenate([[0, 1],
                                  car.position / self.POS_STD,
                                  car.linear_velocity / common_values.CAR_MAX_SPEED,
                                  car.angular_velocity / common_values.CAR_MAX_ANG_VEL,
                                  car.forward(),
                                  car.up(),
                                  [player.boost_amount, player.on_ground, player.has_flip]])
        if self.flatten:
            return np.concatenate([ball_obs, car_obs])
        return np.stack([ball_obs, car_obs])


class AttentionObs(ObsBuilder):
    """
    Observation builder suitable for attention models\n
    Inspired by Necto's obs builder. Missing demo timers, boost timers and boost pad locations\n
    Returns an observation tensor for a player:\n
    8: 1 (player) + 1 (ball) + number of players (default is 6)\n
    \* 32: 4 (player flag, blue team flag, orange team flag, ball flag) + 9 ((relative) normalized position, (relative)
    normalized linear velocity and normalized angular velocity vectors) + 6 (forward and upward rotation axes)
    \+ 4 (boost amount, touching ground flag, has flip flag, is demoed flag) + 8 (previous action)
    \+ 1 (key padding mask used for marking object/player padding in observation)

    The key padding mask is useful in maintaining multiple matches of different sizes and allowing the model
    to play in a variety of settings simultaneously
    """
    # Boost pad locations can be useful for a model trained to pick up and maintain boost
    # Each different sub-model under the large model can potentially use a different part of the observation
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
