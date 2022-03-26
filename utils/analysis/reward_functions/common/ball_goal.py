import numpy as np
from rlgym.utils import common_values, math


def liu_dist_ball2goal(ball_position: np.ndarray, own_goal=False):
    # Inspired by https://arxiv.org/abs/2105.12196
    objective = np.array(common_values.ORANGE_GOAL_BACK) if not own_goal \
        else np.array(common_values.BLUE_GOAL_BACK)

    dist = (np.linalg.norm(ball_position - objective, 2, axis=-1) -
            (common_values.BACK_NET_Y - common_values.BACK_WALL_Y + common_values.BALL_RADIUS))
    return np.exp(-0.5 * dist / common_values.BALL_MAX_SPEED)


def velocity_ball2goal(ball_position, ball_lin_velocity, own_goal=False, use_scalar_projection=False):
    objective = np.array(common_values.ORANGE_GOAL_BACK) if not own_goal \
        else np.array(common_values.BLUE_GOAL_BACK)
    pos_diff = objective - ball_position
    if use_scalar_projection:
        return math.scalar_projection(ball_lin_velocity, pos_diff)
    else:
        norm_pos_diff = pos_diff / np.linalg.norm(pos_diff, 2, axis=-1)[:, None]
        ball_lin_velocity = ball_lin_velocity / common_values.BALL_MAX_SPEED
        return np.dot(norm_pos_diff, ball_lin_velocity)


def ball_y_coord(ball_position, exponent=1):
    """Exponent must be odd so that negative y values produce negative rewards"""
    return (ball_position[:, 1] / (common_values.BACK_WALL_Y + common_values.BALL_RADIUS)) ** exponent
