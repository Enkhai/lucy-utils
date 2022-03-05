import numpy as np
from rlgym.utils import common_values
from rlgym.utils.gamestates import GameState, PlayerData
from rlgym.utils.reward_functions import common_rewards
from rlgym.utils.reward_functions.reward_function import RewardFunction
from rlgym_tools.extra_rewards import diff_reward


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


class DiffPotentialReward(diff_reward.DiffReward):
    """
    Potential-based reward shaping function with a `negative_slope` magnitude parameter
    """

    def __init__(self, reward_function: RewardFunction, gamma=0.99, negative_slope=1.):
        super(DiffPotentialReward, self).__init__(reward_function, negative_slope)
        self.gamma = gamma

    def _calculate_diff(self, player, rew):
        last = self.last_values.get(player.car_id)
        self.last_values[player.car_id] = rew
        if last is not None:
            ret = (self.gamma * rew) - last
            return self.negative_slope * ret if ret < 0 else ret
        else:
            return 0


class LiuDistancePlayerToBall(RewardFunction):
    """
    A natural extension of a "Player close to ball" reward, inspired by https://arxiv.org/abs/2105.12196
    """

    def __init__(self, dispersion=1., density=1.):
        super(LiuDistancePlayerToBall, self).__init__()
        self.dispersion = dispersion
        self.density = density

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        dist = np.linalg.norm(player.car_data.position - state.ball.position) - common_values.BALL_RADIUS
        return np.exp(-0.5 * dist / (common_values.CAR_MAX_SPEED * self.dispersion)) ** (1 / self.density)


class SignedLiuDistanceBallToGoalReward(common_rewards.LiuDistanceBallToGoalReward):
    """
    A natural extension of a "Ball close to target" reward, inspired by https://arxiv.org/abs/2105.12196.\n
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
