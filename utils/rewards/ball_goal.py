import numpy as np
from rlgym.utils import common_values
from rlgym.utils.gamestates import GameState, PlayerData
from rlgym.utils.reward_functions import common_rewards


class BallYCoordinateReward(common_rewards.BallYCoordinateReward):
    """
    Ball distance from goal wall reward that resolves the odd exponent issue
    """

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.team_num == common_values.BLUE_TEAM:
            rew = (state.ball.position[1] / (common_values.BACK_WALL_Y + common_values.BALL_RADIUS))
        else:
            rew = (state.inverted_ball.position[1] / (common_values.BACK_WALL_Y + common_values.BALL_RADIUS))
        rew = (np.abs(rew) ** self.exponent) * np.sign(rew)
        return rew


class LiuDistanceBallToGoalReward(common_rewards.LiuDistanceBallToGoalReward):
    """
    A natural extension of a "Ball close to target" reward, inspired by https://arxiv.org/abs/2105.12196.
    """
    _goal_depth = common_rewards.BACK_NET_Y - common_values.BACK_WALL_Y + common_values.BALL_RADIUS

    def __init__(self, dispersion=1., density=1., own_goal=False):
        super(LiuDistanceBallToGoalReward, self).__init__(own_goal)
        self.dispersion = dispersion
        self.density = density

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.team_num == common_values.BLUE_TEAM and not self.own_goal \
                or player.team_num == common_values.ORANGE_TEAM and self.own_goal:
            objective = np.array(common_values.ORANGE_GOAL_BACK)
        else:
            objective = np.array(common_values.BLUE_GOAL_BACK)

        # Compensate for moving objective to back of net
        dist = np.linalg.norm(state.ball.position - objective) - self._goal_depth
        # with dispersion
        rew = np.exp(-0.5 * dist / (common_values.BALL_MAX_SPEED * self.dispersion))
        # with density
        rew = rew ** (1 / self.density)
        return rew


class SignedLiuDistanceBallToGoalReward(common_rewards.LiuDistanceBallToGoalReward):
    """
    A natural extension of a signed "Ball close to target" reward, inspired by https://arxiv.org/abs/2105.12196.\n
    Produces an approximate reward of 0 at ball position [side_wall, 0, ball_radius].
    """
    _goal_depth = common_rewards.BACK_NET_Y - common_values.BACK_WALL_Y + common_values.BALL_RADIUS
    # trigonometry problem solution - distance normalization factor that helps produce an approximate reward value of 0
    # at ball position [4096, 0, 93]
    _distance_norm = 4570

    def __init__(self, dispersion=1., density=1., own_goal=False):
        super(SignedLiuDistanceBallToGoalReward, self).__init__(own_goal)
        self.dispersion = dispersion
        self.density = density

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.team_num == common_values.BLUE_TEAM and not self.own_goal \
                or player.team_num == common_values.ORANGE_TEAM and self.own_goal:
            objective = np.array(common_values.ORANGE_GOAL_BACK)
        else:
            objective = np.array(common_values.BLUE_GOAL_BACK)

        # Compensate for moving objective to back of net
        dist = np.linalg.norm(state.ball.position - objective) - self._goal_depth
        # with dispersion
        rew = np.exp(-0.5 * dist / (self._distance_norm * self.dispersion))
        # signed
        rew = (rew - 0.5) * 2
        # with density
        rew = (np.abs(rew) ** (1 / self.density)) * np.sign(rew)
        return rew
