from typing import Any

import numpy as np
from rlgym.utils import ObsBuilder
from rlgym.utils.common_values import BOOST_LOCATIONS, BLUE_TEAM, ORANGE_TEAM
from rlgym.utils.gamestates import GameState, PlayerData


# Taken from Necto


class NectoObs(ObsBuilder):
    _boost_locations = np.array(BOOST_LOCATIONS)
    _invert = np.array([1] * 5 + [-1, -1, 1] * 5 + [1] * 4 + [1] * 8 + [1])
    _norm = np.array([1.] * 5 + [2300] * 6 + [1] * 6 + [5.5] * 3 + [1] * 4 + [1] * 8 + [1])

    def __init__(self, n_players=6, tick_skip=8):
        """
        Returns an observation space of shape (1 batch size, 1 `q` player query + 41 `kv` objects, 32 features)

        Features:
         - 0-5 flags: main player, teammate, opponent, ball, boost
         - 5-8 (relative) normalized position
         - 8-11 (relative) normalized linear velocity
         - 11-14 forward vector
         - 14-17 up vector
         - 17-20 normalized angular velocity
         - 20 boost amount
         - 21 boost / demo timer
         - 22 on ground flag
         - 23 has flip flag
         - 24-32 previous action (zeroes for the kv)
         - 32 key padding mask boolean

        :param n_players: Number of players possible for each match
        :param tick_skip: Physics frame skip, default is 8 (120 / 8 = 15 frames per second)
        """
        super().__init__()
        self.n_players = n_players
        self.demo_timers = None
        self.boost_timers = None
        self.current_state = None
        self.current_qkv = None
        self.tick_skip = tick_skip

    def reset(self, initial_state: GameState):
        self.demo_timers = np.zeros(self.n_players)
        self.boost_timers = np.zeros(len(initial_state.boost_pads))
        # self.current_state = initial_state

    def _maybe_update_obs(self, state: GameState):
        if state == self.current_state:  # No need to update
            return

        if self.boost_timers is None:
            self.reset(state)
        else:
            self.current_state = state

        qkv = np.zeros((1 + self.n_players + len(state.boost_pads), 33))  # Ball, players, boosts
        qkv[:, -1] = 1  # key padding mask

        # Add ball
        n = 0
        ball = state.ball
        qkv[0, 3] = 1  # is_ball
        qkv[0, 5:8] = ball.position
        qkv[0, 8:11] = ball.linear_velocity
        qkv[0, 17:20] = ball.angular_velocity
        qkv[0, -1] = 0  # key padding mask

        # Add players
        n += 1
        demos = np.zeros(self.n_players)  # Which players are currently demoed
        for player in state.players:
            if player.team_num == BLUE_TEAM:
                qkv[n, 1] = 1  # is_teammate
            else:
                qkv[n, 2] = 1  # is_opponent
            car_data = player.car_data
            qkv[n, 5:8] = car_data.position
            qkv[n, 8:11] = car_data.linear_velocity
            qkv[n, 11:14] = car_data.forward()
            qkv[n, 14:17] = car_data.up()
            qkv[n, 17:20] = car_data.angular_velocity
            qkv[n, 20] = player.boost_amount
            #             qkv[0, n, 21] = player.is_demoed
            demos[n - 1] = player.is_demoed  # Keep track for demo timer
            qkv[n, 22] = player.on_ground
            qkv[n, 23] = player.has_flip
            qkv[n, -1] = 0  # key padding mask
            n += 1

        # Add boost pads
        n = 1 + self.n_players
        boost_pads = state.boost_pads
        qkv[n:, 4] = 1  # is_boost
        qkv[n:, 5:8] = self._boost_locations
        qkv[n:, 20] = 0.12 + 0.88 * (self._boost_locations[:, 2] > 72)  # Boost amount
        #         qkv[0, n:, 21] = boost_pads
        qkv[n:, -1] = 0  # key padding mask

        # Boost and demo timers
        new_boost_grabs = (boost_pads == 1) & (self.boost_timers == 0)  # New boost grabs since last frame
        self.boost_timers[new_boost_grabs] = 0.4 + 0.6 * (self._boost_locations[new_boost_grabs, 2] > 72)
        self.boost_timers *= boost_pads  # Make sure we have zeros right
        qkv[1 + self.n_players:, 21] = self.boost_timers
        self.boost_timers -= self.tick_skip / 1200  # Pre-normalized, 120 fps for 10 seconds
        self.boost_timers[self.boost_timers < 0] = 0

        new_demos = (demos == 1) & (self.demo_timers == 0)
        self.demo_timers[new_demos] = 0.3
        self.demo_timers *= demos
        qkv[1: 1 + self.n_players, 21] = self.demo_timers
        self.demo_timers -= self.tick_skip / 1200
        self.demo_timers[self.demo_timers < 0] = 0

        # Store results
        self.current_qkv = qkv / self._norm

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:
        # Autodetect of zeroes bad for Lucy  # ++
        # if self.boost_timers is None:  # --
        #     return np.zeros(0)  # Obs space autodetect, make Aech happy  # --
        self._maybe_update_obs(state)
        invert = player.team_num == ORANGE_TEAM

        qkv = self.current_qkv.copy()

        main_n = state.players.index(player) + 1
        qkv[main_n, 0] = 1  # is_main
        if invert:
            qkv[:, (1, 2)] = qkv[:, (2, 1)]  # Swap blue/orange
            qkv *= self._invert  # Negate x and y values

        q = qkv[[main_n], :]
        q[0, -9:-1] = previous_action
        kv = qkv

        # With EARLPerceiver we can use relative coords+vel(+more?) for key/value tensor, might be smart
        kv[:, 5:11] -= q[0, 5:11]
        obs = np.concatenate([q, kv])

        return obs
