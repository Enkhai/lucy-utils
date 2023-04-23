from typing import Any

import numpy as np
from rlgym.utils.common_values import BOOST_LOCATIONS
from rlgym.utils.gamestates import GameState, PlayerData
from rlgym.utils.math import rotation_to_quaternion


# Taken from Necto


def encode_gamestate(state: GameState):
    state_vals = [0, state.blue_score, state.orange_score]
    state_vals += state.boost_pads.tolist()

    for bd in (state.ball, state.inverted_ball):
        state_vals += bd.position.tolist()
        state_vals += bd.linear_velocity.tolist()
        state_vals += bd.angular_velocity.tolist()

    for p in state.players:
        state_vals += [p.car_id, p.team_num]
        for cd in (p.car_data, p.inverted_car_data):
            state_vals += cd.position.tolist()
            state_vals += rotation_to_quaternion(cd.rotation_mtx()).tolist()
            state_vals += cd.linear_velocity.tolist()
            state_vals += cd.angular_velocity.tolist()
        state_vals += [
            0,
            0,
            0,
            0,
            0,
            p.is_demoed,
            p.on_ground,
            p.ball_touched,
            p.has_flip,
            p.boost_amount
        ]
    return state_vals


class BatchedObsBuilder:
    def __init__(self):
        super().__init__()
        self.current_state = None
        self.current_obs = None

    def batched_build_obs(self, encoded_states: np.ndarray) -> Any:
        raise NotImplementedError

    def add_actions(self, obs: Any, previous_actions: np.ndarray, player_index=None):
        # Modify current obs to include action
        # player_index=None means actions for all players should be provided
        raise NotImplementedError

    def _reset(self, initial_state: GameState):
        raise NotImplementedError

    def reset(self, initial_state: GameState):
        self.current_state = False
        self.current_obs = None
        self._reset(initial_state)

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:
        # if state != self.current_state:
        self.current_obs = self.batched_build_obs(
            np.expand_dims(encode_gamestate(state), axis=0)
        )
        self.current_state = state

        for i, p in enumerate(state.players):
            if p == player:
                self.add_actions(self.current_obs, previous_action, i)
                return self.current_obs[i]


IS_SELF, IS_MATE, IS_OPP, IS_BALL, IS_BOOST = range(5)
POS = slice(5, 8)
LIN_VEL = slice(8, 11)
FW = slice(11, 14)
UP = slice(14, 17)
ANG_VEL = slice(17, 20)
BOOST, DEMO, ON_GROUND, HAS_FLIP = range(20, 24)
ACTIONS = slice(24, 32)

BALL_STATE_LENGTH = 18
PLAYER_CAR_STATE_LENGTH = 13
PLAYER_TERTIARY_INFO_LENGTH = 10
PLAYER_INFO_LENGTH = 2 + 2 * PLAYER_CAR_STATE_LENGTH + PLAYER_TERTIARY_INFO_LENGTH


class NextoObsBuilder(BatchedObsBuilder):
    _invert = np.array([1] * 5 + [-1, -1, 1] * 5 + [1] * 4)
    _norm = np.array([1.] * 5 + [2300] * 6 + [1] * 6 + [5.5] * 3 + [1] * 4)

    def __init__(self, field_info=None, n_players=6, tick_skip=8):
        super().__init__()
        self.n_players = n_players
        self.demo_timers = None
        self.boost_timers = None
        self.tick_skip = tick_skip
        if field_info is None:
            self._boost_locations = np.array(BOOST_LOCATIONS)
            self._boost_types = self._boost_locations[:, 2] > 72
        else:
            self._boost_locations = np.array([[bp.location.x, bp.location.y, bp.location.z]
                                              for bp in field_info.boost_pads[:field_info.num_boosts]])
            self._boost_types = np.array([bp.is_full_boost for bp in field_info.boost_pads[:field_info.num_boosts]])

    def _reset(self, initial_state: GameState):
        self.demo_timers = np.zeros(len(initial_state.players))
        self.boost_timers = np.zeros(len(initial_state.boost_pads))

    @staticmethod
    def _quats_to_rot_mtx(quats: np.ndarray) -> np.ndarray:
        # From rlgym.utils.math.quat_to_rot_mtx
        w = -quats[:, 0]
        x = -quats[:, 1]
        y = -quats[:, 2]
        z = -quats[:, 3]

        theta = np.zeros((quats.shape[0], 3, 3))

        norm = np.einsum("fq,fq->f", quats, quats)

        sel = norm != 0

        w = w[sel]
        x = x[sel]
        y = y[sel]
        z = z[sel]

        s = 1.0 / norm[sel]

        # front direction
        theta[sel, 0, 0] = 1.0 - 2.0 * s * (y * y + z * z)
        theta[sel, 1, 0] = 2.0 * s * (x * y + z * w)
        theta[sel, 2, 0] = 2.0 * s * (x * z - y * w)

        # left direction
        theta[sel, 0, 1] = 2.0 * s * (x * y - z * w)
        theta[sel, 1, 1] = 1.0 - 2.0 * s * (x * x + z * z)
        theta[sel, 2, 1] = 2.0 * s * (y * z + x * w)

        # up direction
        theta[sel, 0, 2] = 2.0 * s * (x * z + y * w)
        theta[sel, 1, 2] = 2.0 * s * (y * z - x * w)
        theta[sel, 2, 2] = 1.0 - 2.0 * s * (x * x + y * y)

        return theta

    @staticmethod
    def convert_to_relative(q, kv):
        # kv[..., POS.start:LIN_VEL.stop] -= q[..., POS.start:LIN_VEL.stop]
        kv[..., POS] -= q[..., POS]
        forward = q[..., FW]
        theta = np.arctan2(forward[..., 0], forward[..., 1])
        theta = np.expand_dims(theta, axis=-1)
        ct = np.cos(theta)
        st = np.sin(theta)
        xs = kv[..., POS.start:ANG_VEL.stop:3]
        ys = kv[..., POS.start + 1:ANG_VEL.stop:3]
        # Use temp variables to prevent modifying original array
        nx = ct * xs - st * ys
        ny = st * xs + ct * ys
        kv[..., POS.start:ANG_VEL.stop:3] = nx  # x-components
        kv[..., POS.start + 1:ANG_VEL.stop:3] = ny  # y-components

    def batched_build_obs(self, encoded_states: np.ndarray):
        ball_start_index = 3 + len(self._boost_locations)
        players_start_index = ball_start_index + BALL_STATE_LENGTH
        player_length = PLAYER_INFO_LENGTH

        n_players = (encoded_states.shape[1] - players_start_index) // player_length
        lim_players = n_players if self.n_players is None else self.n_players
        n_entities = lim_players + 1 + 34

        # SELECTORS
        sel_players = slice(0, lim_players)
        sel_ball = sel_players.stop
        sel_boosts = slice(sel_ball + 1, None)

        # MAIN ARRAYS
        q = np.zeros((n_players, encoded_states.shape[0], 1, 32))
        kv = np.zeros((n_players, encoded_states.shape[0], n_entities, 24))  # Keys and values are (mostly) shared
        m = np.zeros((n_players, encoded_states.shape[0], n_entities))  # Mask is shared

        # BALL
        kv[:, :, sel_ball, 3] = 1
        kv[:, :, sel_ball, np.r_[POS, LIN_VEL, ANG_VEL]] = encoded_states[:, ball_start_index: ball_start_index + 9]

        # BOOSTS
        kv[:, :, sel_boosts, IS_BOOST] = 1
        kv[:, :, sel_boosts, POS] = self._boost_locations
        kv[:, :, sel_boosts, BOOST] = 0.12 + 0.88 * (self._boost_locations[:, 2] > 72)
        kv[:, :, sel_boosts, DEMO] = encoded_states[:, 3:3 + 34]  # FIXME boost timer

        # PLAYERS
        teams = encoded_states[0, players_start_index + 1::player_length]
        kv[:, :, :n_players, IS_MATE] = 1 - teams  # Default team is blue
        kv[:, :, :n_players, IS_OPP] = teams
        for i in range(n_players):
            encoded_player = encoded_states[:,
                             players_start_index + i * player_length: players_start_index + (i + 1) * player_length]

            kv[i, :, i, IS_SELF] = 1
            kv[:, :, i, POS] = encoded_player[:, 2: 5]  # TODO constants for these indices
            kv[:, :, i, LIN_VEL] = encoded_player[:, 9: 12]
            quats = encoded_player[:, 5: 9]
            rot_mtx = self._quats_to_rot_mtx(quats)
            kv[:, :, i, FW] = rot_mtx[:, :, 0]
            kv[:, :, i, UP] = rot_mtx[:, :, 2]
            kv[:, :, i, ANG_VEL] = encoded_player[:, 12: 15]
            kv[:, :, i, BOOST] = encoded_player[:, 37]
            kv[:, :, i, DEMO] = encoded_player[:, 33]  # FIXME demo timer
            kv[:, :, i, ON_GROUND] = encoded_player[:, 34]
            kv[:, :, i, HAS_FLIP] = encoded_player[:, 36]

        kv[teams == 1] *= self._invert
        kv[np.argwhere(teams == 1), ..., (IS_MATE, IS_OPP)] = kv[
            np.argwhere(teams == 1), ..., (IS_OPP, IS_MATE)]  # Swap teams

        kv /= self._norm

        for i in range(n_players):
            q[i, :, 0, :kv.shape[-1]] = kv[i, :, i, :]

        self.convert_to_relative(q, kv)
        # kv[:, :, :, 5:11] -= q[:, :, :, 5:11]

        # MASK
        m[:, :, n_players: lim_players] = 1

        # +++ START: custom code +++

        q_ = np.append(q.squeeze(1), np.ones((q.shape[0], q.shape[2], 1)), -1)
        kv_ = np.append(kv.squeeze(1), np.zeros((kv.shape[0], kv.shape[2], 8)), -1)
        obs_ = np.append(kv_, m.squeeze(1)[..., None], -1)
        obs_ = np.append(q_, obs_, 1)

        return [obs_[i] for i in range(n_players)]

        # +++ END: custom code +++

    def add_actions(self, obs: Any, previous_actions: np.ndarray, player_index=None):
        if player_index is None:
            for o, act in zip(obs, previous_actions):
                o[0, ACTIONS] = act
        else:
            obs[player_index, 0, ACTIONS] = previous_actions
