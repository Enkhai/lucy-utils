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
    Rewards the player to goal distance when a winning goal happens.
    """

    def __init__(self, dispersion=0.5, density=1.):
        self.dispersion = dispersion
        self.density = density
        self.n_goals = {}

    def reset(self, initial_state: GameState):
        for player in initial_state.players:
            if player.team_num == common_values.BLUE_TEAM:
                self.n_goals[player.car_id] = initial_state.blue_score
            else:
                self.n_goals[player.car_id] = initial_state.orange_score

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.team_num == common_values.BLUE_TEAM:
            goal_diff = state.blue_score - self.n_goals[player.car_id]
            objective = np.array(common_values.ORANGE_GOAL_BACK)
        else:
            goal_diff = state.orange_score - self.n_goals[player.car_id]
            objective = np.array(common_values.BLUE_GOAL_BACK)
        rew = 0
        if goal_diff > 0:
            player2goal_dist = np.linalg.norm(player.car_data.position - objective) - goal_depth
            rew = np.exp(-0.5 * player2goal_dist / (common_values.CAR_MAX_SPEED * self.dispersion)) ** self.density

            self.n_goals[player.car_id] += goal_diff
        return rew


class DefensivePressureReward(RewardFunction):
    """
    Rewards the inverse player to goal distance when a conceding goal happens.
    Suitable for applying negative rewards to.
    """

    def __init__(self, dispersion=0.5, density=1.):
        self.dispersion = dispersion
        self.density = density
        self.n_concedes = {}

    def reset(self, initial_state: GameState):
        for player in initial_state.players:
            if player.team_num == common_values.BLUE_TEAM:
                self.n_concedes[player.car_id] = initial_state.orange_score
            else:
                self.n_concedes[player.car_id] = initial_state.blue_score

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.team_num == common_values.BLUE_TEAM:
            concede_diff = state.orange_score - self.n_concedes[player.car_id]
            objective = np.array(common_values.BLUE_GOAL_BACK)
        else:
            concede_diff = state.blue_score - self.n_concedes[player.car_id]
            objective = np.array(common_values.ORANGE_GOAL_BACK)
        rew = 0
        if concede_diff > 0:
            player2goal_dist = np.linalg.norm(player.car_data.position - objective) - goal_depth
            rew = 1 - (np.exp(-0.5 * player2goal_dist /
                              (common_values.CAR_MAX_SPEED * self.dispersion)) ** self.density)

            self.n_concedes[player.car_id] += concede_diff
        return rew
