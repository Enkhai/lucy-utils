from typing import Union

import numpy as np
from rlgym.utils import RewardFunction, common_values
from rlgym.utils.gamestates import GameState, PlayerData
from rlgym.utils.reward_functions import common_rewards


class OffensivePotentialReward(RewardFunction):
    """
    Offensive potential function. When the player to ball and ball to goal vectors align
    we should reward player to ball velocity.\n
    Uses a combination of `AlignBallGoal`,`VelocityPlayerToBallReward` and `LiuDistancePlayerToBallReward` rewards.
    """

    def __init__(self, defense=0.5, offense=0.5, dispersion=1, density=1):
        super(OffensivePotentialReward, self).__init__()
        self.align_ball_goal = common_rewards.AlignBallGoal(defense=defense, offense=offense)
        self.velocity_player2ball = common_rewards.VelocityPlayerToBallReward()
        self.liu_dist_player2ball = LiuDistancePlayerToBallReward(dispersion, density)

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        align_ball_rew = self.align_ball_goal.get_reward(player, state, previous_action)
        velocity_player2ball_rew = self.velocity_player2ball.get_reward(player, state, previous_action)
        liu_dist_player2ball_rew = self.liu_dist_player2ball.get_reward(player, state, previous_action)

        # logical AND
        # when both alignment and player to ball velocity are negative we must get a negative output
        sign = ((align_ball_rew >= 0 and velocity_player2ball_rew >= 0) - 0.5) * 2
        # liu_dist_player2ball_rew is positive only, no need to compute for sign
        rew = align_ball_rew * velocity_player2ball_rew * liu_dist_player2ball_rew
        # cube root because we multiply three values between -1 and 1
        # "weighted" product (n_1 * n_2 * ... * n_N) ^ (1 / N)
        return (abs(rew) ** (1 / 3)) * sign


class LiuDistancePlayerToBallReward(RewardFunction):
    """
    A natural extension of a "Player close to ball" reward, inspired by https://arxiv.org/abs/2105.12196
    """

    def __init__(self, dispersion=1., density=1.):
        super(LiuDistancePlayerToBallReward, self).__init__()
        self.dispersion = dispersion
        self.density = density

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        dist = np.linalg.norm(player.car_data.position - state.ball.position) - common_values.BALL_RADIUS
        return np.exp(-0.5 * dist / (common_values.CAR_MAX_SPEED * self.dispersion)) ** (1 / self.density)


class TouchBallToGoalAccelerationReward(RewardFunction):
    """
    An extension of the "Touch aerial acceleration" reward used in Necto.

    Triggers when the player touches the ball. Composite reward of two components:
     - Ball height reward, ranges between 0 and 1
     - Ball to goal acceleration reward. Signed reward function. Rewards a value of 1 for 2300 units/sec of ball to goal
     acceleration (standing still to supersonic)
    """

    def __init__(self, aerial_weight=1., accel_weight=1, own_goal=False):
        super(TouchBallToGoalAccelerationReward, self).__init__()
        self.aerial_weight = aerial_weight
        self.accel_weight = accel_weight
        self.last_state: Union[GameState, None] = None
        self.own_goal = own_goal

    def reset(self, initial_state: GameState):
        self.last_state = None

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        rew = 0
        if player.ball_touched:
            if player.team_num == common_values.BLUE_TEAM and not self.own_goal \
                    or player.team_num == common_values.ORANGE_TEAM and self.own_goal:
                objective = np.array(common_values.ORANGE_GOAL_BACK)
            else:
                objective = np.array(common_values.BLUE_GOAL_BACK)

            last_ball2goal_vel = 0
            if self.last_state:
                last_ball2goal_pos_diff = objective - self.last_state.ball.position
                last_ball2goal_vel = np.dot(last_ball2goal_pos_diff / np.linalg.norm(last_ball2goal_pos_diff),
                                            self.last_state.ball.linear_velocity)

            ball_pos = state.ball.position
            ball2goal_pos_diff = objective - ball_pos
            ball2goal_vel = np.dot(ball2goal_pos_diff / np.linalg.norm(ball2goal_pos_diff),
                                   state.ball.linear_velocity)

            rew = (self.aerial_weight * ball_pos[2] / common_values.CEILING_Z +
                   self.accel_weight * (ball2goal_vel - last_ball2goal_vel) / common_values.CAR_MAX_SPEED)

        self.last_state = state
        return rew
