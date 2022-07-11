from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from rlgym.utils import RewardFunction, common_values
from rlgym.utils.gamestates import GameState, PlayerData

from ..rewards._common import goal_depth


class PressureReward(RewardFunction, ABC):
    """
    Rewards the discounted mean pressure when a condition is fulfilled. After the condition is met,
    pressure is released and, if pressure is applicable, the timer starts over.

    Pressure is computed as such: 0.5 + 0.5 * ((number of allies pressing / number of allies) -
    (number of opponents pressing / number of opponents))

    The pressure zone is defined as the zone within threshold distance from the goal.
    Pressing players are considered players within the pressure zone.

    Mean pressure is computed for the number of frames the ball lies within the pressure zone.

    The reward is additionally discounted for the number of frames the ball has been within the pressure zone
    using a linear function and a cutoff at `cutoff_frame` frame. Furthermore, the shape of the function is
    controlled using an exponent factor.
    """

    _blue_goal = np.array(common_values.BLUE_GOAL_BACK)
    _orange_goal = np.array(common_values.ORANGE_GOAL_BACK)

    def __init__(self,
                 cutoff_frame: int,
                 exponent: float,
                 distance_threshold: Union[float, int],
                 offense: bool,
                 inverted: bool):
        """
        :param cutoff_frame: The frame from which pressure becomes zero. Dictates the negative slop of the decay linear
         function.
        :param exponent: Controls the concavity of the decay function.
        :param distance_threshold: Controls the size of the pressure zone. The pressure zone is defined as the zone
         within threshold distance from the goal back plus the depth (973) of the net and the ball radius.
        :param offense: If True, pressure is rewarded for offense. If False, pressure is rewarded for defense.
        :param inverted: If True, the team for which the pressure applies is rewarded `1 - pressure`. Suitable for
         negative pressure rewards.
        """
        self.cutoff_frame = cutoff_frame
        self.exponent = exponent
        self.distance_threshold = distance_threshold
        self.offense = offense
        self.inverted = inverted
        self.inverted_team = common_values.BLUE_TEAM if inverted else common_values.ORANGE_TEAM
        self.timer = 0
        self.pressure_sum = 0
        self.pressure_team: Union[None, int] = None
        self.last_state: Union[None, GameState] = None

    @abstractmethod
    def _reset(self, state: GameState, is_state_initial=False):
        self.timer = 0
        self.pressure_sum = 0
        self.pressure_team = None
        self.last_state = None if is_state_initial else state

    def reset(self, initial_state: GameState):
        self._reset(initial_state, True)

    def _compute_pressure(self, state: GameState, objective: np.ndarray):
        blue_positions = []
        orange_positions = []
        for p in state.players:
            if p.team_num == common_values.BLUE_TEAM:
                blue_positions.append(p.car_data.position)
            else:
                orange_positions.append(p.car_data.position)
        blue_positions = np.array(blue_positions)
        orange_positions = np.array(orange_positions)

        blue2objective_dist = np.linalg.norm(objective - blue_positions, axis=-1) - goal_depth
        orange2objective_dist = np.linalg.norm(objective - orange_positions, axis=-1) - goal_depth

        blue_pressing = (blue2objective_dist < self.distance_threshold).sum()
        orange_pressing = (orange2objective_dist < self.distance_threshold).sum()

        pressure = 0.5 + 0.5 * ((blue_pressing / blue_positions.shape[0]) -
                                (orange_pressing / orange_positions.shape[0]))
        return pressure

    def _update_pressure(self, state: GameState):
        ball2blue_dist = np.linalg.norm(self._blue_goal - state.ball.position) - goal_depth
        ball2orange_dist = np.linalg.norm(self._orange_goal - state.ball.position) - goal_depth

        if ball2blue_dist < self.distance_threshold:
            objective = self._blue_goal
            self.pressure_team = common_values.ORANGE_TEAM if self.offense else common_values.BLUE_TEAM
        elif ball2orange_dist < self.distance_threshold:
            objective = self._orange_goal
            self.pressure_team = common_values.BLUE_TEAM if self.offense else common_values.ORANGE_TEAM
        else:
            self._reset(state)
            return

        self.pressure_sum += self._compute_pressure(state, objective)
        self.timer += 1

        self.last_state = state

    @abstractmethod
    def condition(self, state: GameState) -> bool:
        """
        The condition should cover the general pressure release case, not depend on the current player and
        be applicable for the entire duration of the state.
        """
        raise NotImplementedError

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        rew = 0

        if self.condition(state):
            if player.team_num == self.pressure_team:
                pressure = self.pressure_sum / self.timer
                if player.team_num == self.inverted_team:
                    pressure = 1 - pressure
                t = min(self.timer - 1, self.cutoff_frame)  # discount from 2nd frame
                discount = (-t / self.cutoff_frame + 1) ** self.exponent
                rew = discount * pressure
            if player == state.players[-1]:
                self._reset(state)  # reset once we reach the last player
            return rew

        if state != self.last_state:
            self._update_pressure(state)

        return rew


class OffensivePressureReward(PressureReward):
    """
    Rewards the discounted mean pressure when a winning goal is scored.

    Pressure is computed as such: 0.5 + 0.5 * ((number of allies offending / number of allies) -
    (number of opponents defending / number of opponents))

    The pressure zone is defined as the zone within threshold distance from the opponent goal.
    Offending and defending players are considered players within the pressure zone.

    Mean pressure is computed for the number of frames the ball lies within the pressure zone.

    The reward is additionally discounted for the number of frames the ball has been within the pressure zone
    using a linear function and a cutoff at `cutoff_frame` frame. Furthermore, the shape of the function is
    controlled using an exponent factor.

    Offensive pressure should be rewarded positively and used to enhance the reward signal when scoring a goal.
    """

    def __init__(self, cutoff_frame=90, exponent=0.7, distance_threshold=3680):
        super(OffensivePressureReward, self).__init__(cutoff_frame, exponent, distance_threshold, True, False)
        self.n_goals = [0, 0]

    def _reset(self, state: GameState, is_state_initial=False):
        super(OffensivePressureReward, self)._reset(state, is_state_initial)
        self.n_goals = [state.blue_score, state.orange_score]

    def condition(self, state: GameState) -> bool:
        if self.pressure_team is None:
            return False
        n_goals = state.blue_score if self.pressure_team == common_values.BLUE_TEAM else state.orange_score
        return n_goals > self.n_goals[self.pressure_team]


class DefensivePressureReward(PressureReward):
    """
    Rewards the discounted mean pressure when a conceding goal is scored.

    Pressure is computed as such: 0.5 + 0.5 * ((number of opponents offending / number of opponents) -
    (number of allies defending / number of allies))

    The pressure zone is defined as the zone within threshold distance from the team goal.
    Offending and defending players are considered players within the pressure zone.

    Mean pressure is computed for the number of frames the ball lies within the pressure zone.

    The reward is additionally discounted for the number of frames the ball has been within the pressure zone
    using a linear function and a cutoff at `cutoff_frame` frame. Furthermore, the shape of the function is
    controlled using an exponent factor.

    Defensive pressure should be rewarded negatively and used to enhance the reward signal when receiving a goal.
    """

    def __init__(self, cutoff_frame=90, exponent=0.7, distance_threshold=3680):
        super(DefensivePressureReward, self).__init__(cutoff_frame, exponent, distance_threshold, False, True)
        self.n_concedes = [0, 0]

    def _reset(self, state: GameState, is_state_initial=False):
        super(DefensivePressureReward, self)._reset(state, is_state_initial)
        self.n_concedes = [state.orange_score, state.blue_score]

    def condition(self, state: GameState) -> bool:
        if self.pressure_team is None:
            return False
        n_concedes = state.orange_score if self.pressure_team == common_values.BLUE_TEAM else state.blue_score
        return n_concedes > self.n_concedes[self.pressure_team]


class CounterPressureReward(PressureReward):
    """
    Rewards the discounted mean pressure when the ball leaves the pressure zone.

    Pressure is computed as such: 0.5 + 0.5 * ((number of allies defending / number of allies) -
    (number of opponents offending / number of opponents))

    The pressure zone is defined as the zone within threshold distance from the team goal.
    Defending and offending players are considered players within the pressure zone.

    Mean pressure is computed for the number of frames the ball lies within the pressure zone.

    The reward is additionally discounted for the number of frames the ball has been within the pressure zone
    using a linear function and a cutoff at `cutoff_frame` frame. Furthermore, the shape of the function is
    controlled using an exponent factor.

    Counter pressure is rewarded only after the ball has existed within the pressure zone for more than
    `frame_threshold` frames in order to avoid instantaneous defensive rewarding.

    Counter pressure should be rewarded positively.
    """

    def __init__(self, cutoff_frame=90, exponent=0.7, distance_threshold=3680, frame_threshold=3):
        super(CounterPressureReward, self).__init__(cutoff_frame, exponent, distance_threshold, False, False)
        self.frame_threshold = frame_threshold

    def _reset(self, state: GameState, is_state_initial=False):
        super(CounterPressureReward, self)._reset(state, is_state_initial)

    def condition(self, state: GameState) -> bool:
        if self.timer < self.frame_threshold:
            return False
        if (state.blue_score > self.last_state.blue_score or
                state.orange_score > self.last_state.orange_score):  # avoid reward on goal
            return False

        objective = self._blue_goal if self.pressure_team == common_values.BLUE_TEAM else self._orange_goal

        last_ball2objective_dist = np.linalg.norm(objective - self.last_state.ball.position) - goal_depth
        ball2objective_dist = np.linalg.norm(objective - state.ball.position) - goal_depth

        return ((last_ball2objective_dist < self.distance_threshold) and
                (ball2objective_dist >= self.distance_threshold))
