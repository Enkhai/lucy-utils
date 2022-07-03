from typing import Union

import numpy as np
from rlgym.utils import RewardFunction, common_values
from rlgym.utils.gamestates import GameState, PlayerData
from rlgym.utils.reward_functions import common_rewards

from ..rewards._common import goal_depth
from ..rewards.player_ball import LiuDistancePlayerToBallReward


class DistanceWeightedAlignBallGoal(RewardFunction):

    def __init__(self, defense=0.5, offense=0.5, dispersion=1., density=1.):
        super(DistanceWeightedAlignBallGoal, self).__init__()
        self.align_ball_goal = common_rewards.AlignBallGoal(defense=defense, offense=offense)
        self.liu_dist_player2ball = LiuDistancePlayerToBallReward(dispersion, density)

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        align_ball_rew = self.align_ball_goal.get_reward(player, state, previous_action)
        liu_dist_player2ball_rew = self.liu_dist_player2ball.get_reward(player, state, previous_action)

        rew = align_ball_rew * liu_dist_player2ball_rew
        # square root because we multiply two values between -1 and 1
        # "weighted" product (n_1 * n_2 * ... * n_N) ^ (1 / N)
        return np.sqrt(abs(rew)) * np.sign(rew)


class EventReward(common_rewards.EventReward):
    """
    An extension of the EventReward function that adds a `demoed` reward
    """

    def __init__(self, goal=0., team_goal=0., concede=-0., touch=0., shot=0., save=0., demo=0., demoed=0.):
        """
        :param goal: reward for goal scored by player.
        :param team_goal: reward for goal scored by player's team.
        :param concede: reward for goal scored by opponents. Should be negative if used as punishment.
        :param touch: reward for touching the ball.
        :param shot: reward for shooting the ball (as detected by Rocket League).
        :param save: reward for saving the ball (as detected by Rocket League).
        :param demo: reward for demolishing a player.
        :param demoed: reward for being demolished by another player. Should be negative if used as punishment.
        """
        super().__init__()
        self.weights = np.array([goal, team_goal, concede, touch, shot, save, demo, demoed])

        # Need to keep track of last registered value to detect changes
        self.last_registered_values = {}

    @staticmethod
    def _extract_values(player: PlayerData, state: GameState):
        if player.team_num == common_values.BLUE_TEAM:
            team, opponent = state.blue_score, state.orange_score
        else:
            team, opponent = state.orange_score, state.blue_score

        return np.array([player.match_goals, team, opponent, player.ball_touched, player.match_shots,
                         player.match_saves, player.match_demolishes, player.is_demoed])


class OffensivePressureReward(RewardFunction):
    """
    Rewards the discounted mean pressure when a winning goal is scored.

    Pressure is computed as such: 0.5 - 0.5 * ((number of allies offending / number of allies) -
    (number of opponents defending / number of opponents))

    Offending and defending players must be within threshold distance from the goal.

    Mean pressure is computed for the number of frames the ball lies within threshold distance from the opponent goal.

    The reward is halved when the ball has been in threshold distance for a number of `half_life_frames` frames.
    """

    # TODO: compute suitable half life frames
    def __init__(self, half_life_frames=38, distance_threshold=3680):
        self.half_life_frames = half_life_frames
        self.gamma = np.exp(np.log(0.5) / half_life_frames)
        self.distance_threshold = distance_threshold
        self.timer = 0
        self.pressure_sums = {}
        self.n_goals = {}
        self.last_state: Union[None, GameState] = None

    def reset(self, initial_state: GameState):
        self.timer = 0
        for player in initial_state.players:
            self.pressure_sums[player.car_id] = 0
            if player.team_num == common_values.BLUE_TEAM:
                self.n_goals[player.car_id] = initial_state.blue_score
            else:
                self.n_goals[player.car_id] = initial_state.orange_score
        self.last_state = None

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        n_goals = state.blue_score if player.team_num == common_values.BLUE_TEAM else state.orange_score

        if n_goals > self.n_goals[player.car_id]:
            rew = (self.gamma ** self.timer) * (self.pressure_sums[player.car_id] / self.timer)
            self.pressure_sums = {p.car_id: 0 for p in state.players}
            self.timer = 0
            return rew

        if state != self.last_state:
            self._update_pressure(state)

        return 0

    def _update_pressure(self, state: GameState):
        blue_goal = np.array(common_rewards.BLUE_GOAL_BACK)
        orange_goal = np.array(common_rewards.ORANGE_GOAL_BACK)

        ball2blue_dist = np.linalg.norm(blue_goal - state.ball.position) - goal_depth
        ball2orange_dist = np.linalg.norm(orange_goal - state.ball.position) - goal_depth

        if ball2blue_dist < self.distance_threshold:
            objective = blue_goal
            attacker = common_values.ORANGE_TEAM
        elif ball2orange_dist < self.distance_threshold:
            objective = orange_goal
            attacker = common_values.BLUE_TEAM
        else:
            self.pressure_sums = {p.car_id: 0 for p in state.players}
            self.timer = 0
            return

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

        if attacker == common_values.BLUE_TEAM:
            pressure = 0.5 - 0.5 * ((orange_pressing / orange_positions.shape[0]) -
                                    (blue_pressing / blue_positions.shape[0]))
        else:
            pressure = 0.5 - 0.5 * ((blue_pressing / blue_positions.shape[0]) -
                                    (orange_pressing / orange_positions.shape[0]))

        for p in state.players:
            if p.team_num == attacker:
                self.pressure_sums[p.car_id] = pressure
            else:
                self.pressure_sums[p.car_id] = 0

        self.timer += 1


class DefensivePressureReward(RewardFunction):
    """
    Rewards the discounted mean pressure when a conceding goal is avoided.

    Pressure is computed as such: 0.5 - 0.5 * ((number of allies defending / number of allies) -
    (number of opponents offending / number of opponents))

    Offending and defending players must be within threshold distance from the goal.

    Mean pressure is computed for the number of frames the ball lies within threshold distance from the team goal.

    The reward is halved when the ball has been in threshold distance for a number of `half_life_frames` frames.
    """

    # TODO: compute suitable half life frames
    def __init__(self, half_life_frames=38, distance_threshold=3680):
        self.half_life_frames = half_life_frames
        self.gamma = np.exp(np.log(0.5) / half_life_frames)
        self.distance_threshold = distance_threshold
        self.timer = 0
        self.pressure_sums = {}
        self.n_concedes = {}
        self.last_state: Union[None, GameState] = None

    def reset(self, initial_state: GameState):
        self.timer = 0
        for player in initial_state.players:
            self.pressure_sums[player.car_id] = 0
            if player.team_num == common_values.BLUE_TEAM:
                self.n_concedes[player.car_id] = initial_state.orange_score
            else:
                self.n_concedes[player.car_id] = initial_state.blue_score
        self.last_state = None

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # TODO: implement this
        return 0
