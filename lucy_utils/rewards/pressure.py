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

    Pressure is computed as such: 0.5 - 0.5 * ((number of allies pressing / number of allies) -
    (number of opponents pressing / number of opponents))

    The pressure zone is defined as the zone within threshold distance from the goal.
    Pressing players are considered players within the pressure zone.

    Mean pressure is computed for the number of frames the ball lies within the pressure zone.

    The reward is additionally discounted for the number of frames the ball has been within the pressure zone
    and halved when `half_life_frames` frames have passed.
    """

    _blue_goal = np.array(common_values.BLUE_GOAL_BACK)
    _orange_goal = np.array(common_values.ORANGE_GOAL_BACK)

    def __init__(self, half_life_frames: int, distance_threshold: Union[float, int], offense: bool):
        self.half_life_frames = half_life_frames
        self.gamma = np.exp(np.log(0.5) / (half_life_frames - 1))
        self.distance_threshold = distance_threshold
        self.offense = offense
        self.timer = 0
        self.pressure_sum = 0
        self.pressure_team: Union[None, int] = None
        self.last_state: Union[None, GameState] = None

    @abstractmethod
    def _reset(self, state: Union[None, GameState]):
        self.timer = 0
        self.pressure_sum = 0
        self.pressure_team = None
        self.last_state = state

    def reset(self, initial_state: GameState):
        self._reset(None)

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
    def condition(self, player: PlayerData, state: GameState) -> bool:
        raise NotImplementedError

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        rew = 0

        if self.condition(player, state):
            if player.team_num == self.pressure_team:
                pressure = self.pressure_sum / self.timer
                if player.team_num == common_values.ORANGE_TEAM:
                    pressure = 1 - pressure
                rew = (self.gamma ** (self.timer - 1)) * pressure
            self._reset(state)
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
    and halved when `half_life_frames` frames have passed.
    """

    # TODO: compute appropriate half life frames
    def __init__(self, half_life_frames=38, distance_threshold=3680):
        super(OffensivePressureReward, self).__init__(half_life_frames, distance_threshold, True)
        self.n_goals = {}

    def _reset(self, state: GameState):
        super(OffensivePressureReward, self)._reset(state)
        for p in state.players:
            self.n_goals[p.car_id] = state.blue_score if p.team_num == common_values.BLUE_TEAM else state.orange_score

    def condition(self, player: PlayerData, state: GameState) -> bool:
        n_goals = state.blue_score if player.team_num == common_values.BLUE_TEAM else state.orange_score
        return n_goals > self.n_goals[player.car_id]


class DefensivePressureReward(PressureReward):
    """
    Rewards the discounted mean pressure when the ball leaves the pressure zone.

    Pressure is computed as such: 0.5 + 0.5 * ((number of allies defending / number of allies) -
    (number of opponents offending / number of opponents))

    The pressure zone is defined as the zone within threshold distance from the team goal.
    Defending and offending players are considered players within the pressure zone.

    Mean pressure is computed for the number of frames the ball lies within the pressure zone.

    The reward is additionally discounted for the number of frames the ball has been within the pressure zone
    and halved when `half_life_frames` frames have passed.
    """

    # TODO: compute appropriate half life frames
    def __init__(self, half_life_frames=38, distance_threshold=3680):
        super(DefensivePressureReward, self).__init__(half_life_frames, distance_threshold, False)

    def _reset(self, state: GameState):
        super(DefensivePressureReward, self)._reset(state)

    def condition(self, player: PlayerData, state: GameState) -> bool:
        objective = self._blue_goal if player.team_num == common_values.BLUE_TEAM else self._orange_goal

        last_ball2objective_dist = np.linalg.norm(objective - self.last_state.ball.position) - goal_depth
        ball2objective_dist = np.linalg.norm(objective - state.ball.position) - goal_depth

        return ((last_ball2objective_dist < self.distance_threshold) and
                (ball2objective_dist >= self.distance_threshold))
