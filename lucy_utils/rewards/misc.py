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
        self.pressure_sum = 0
        self.n_goals = {}

    def reset(self, initial_state: GameState):
        for player in initial_state.players:
            if player.team_num == common_values.BLUE_TEAM:
                self.n_goals[player.car_id] = initial_state.blue_score
            else:
                self.n_goals[player.car_id] = initial_state.orange_score

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.team_num == common_values.BLUE_TEAM:
            objective = np.array(common_values.ORANGE_GOAL_BACK)
            n_goals = state.blue_score
        else:
            objective = np.array(common_values.BLUE_GOAL_BACK)
            n_goals = state.orange_score

        if n_goals > self.n_goals[player.car_id]:
            rew = (self.gamma ** self.timer) * (self.pressure_sum / self.timer)
            self.pressure_sum = 0
            self.timer = 0
            return rew

        ball2goal_dist = np.linalg.norm(objective - state.ball.position) - goal_depth
        if ball2goal_dist < self.distance_threshold:
            ally_positions = []
            for p in state.players:
                if p.team_num == player.team_num:
                    ally_positions.append(p.car_data.position)
            ally_positions = np.array(ally_positions)

            ally2goal_dist = np.linalg.norm(objective - ally_positions, axis=-1) - goal_depth
            n_ally_pressing = (ally2goal_dist < self.distance_threshold).sum()

            enemy_positions = []
            for p in state.players:
                if p.team_num != player.team_num:
                    enemy_positions.append(p.car_data.position)
            enemy_positions = np.array(enemy_positions)

            enemy2goal_dist = np.linalg.norm(objective - enemy_positions, axis=-1) - goal_depth
            n_enemy_pressing = (enemy2goal_dist < self.distance_threshold).sum()

            pressure = 0.5 - 0.5 * ((n_ally_pressing / ally_positions.shape[0]) -
                                    (n_enemy_pressing / enemy_positions.shape[0]))
            self.pressure_sum += pressure
            self.timer += 1
        else:
            self.pressure_sum = 0
            self.timer = 0

        return 0


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
        self.pressure_sum = 0
        self.n_concedes = {}

    def reset(self, initial_state: GameState):
        for player in initial_state.players:
            if player.team_num == common_values.BLUE_TEAM:
                self.n_concedes[player.car_id] = initial_state.orange_score
            else:
                self.n_concedes[player.car_id] = initial_state.blue_score

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # TODO: implement this
        return 0
