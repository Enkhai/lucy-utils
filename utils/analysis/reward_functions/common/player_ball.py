import numpy as np
from rlgym.utils import common_values, math


def liu_dist_player2ball(player_position, ball_position):
    # Inspired by https://arxiv.org/abs/2105.12196
    dist = np.linalg.norm(player_position - ball_position, 2) - common_values.BALL_RADIUS
    return np.exp(-0.5 * dist / common_values.CAR_MAX_SPEED)


def velocity_player2ball(player_position,
                         player_lin_velocity,
                         ball_position,
                         use_scalar_projection=False):
    # """Uses minimum negative default position for the ball"""
    # np.array([-common_values.SIDE_WALL_X + common_values.BALL_RADIUS,
    #           -common_values.BACK_WALL_Y + common_values.BALL_RADIUS,
    #           0 + common_values.BALL_RADIUS])

    pos_diff = ball_position - player_position
    if use_scalar_projection:
        return math.scalar_projection(player_lin_velocity, pos_diff)
    else:
        norm_pos_dif = pos_diff / np.linalg.norm(pos_diff, 2)
        player_lin_velocity /= common_values.CAR_MAX_SPEED
        return float(np.dot(norm_pos_dif, player_lin_velocity))


def face_ball(player_position, ball_position, player_forward_vec):
    pos_diff = ball_position - player_position
    norm_pos_diff = pos_diff / np.linalg.norm(pos_diff, 2)
    return float(np.dot(player_forward_vec, norm_pos_diff))


def touch_ball(ball_position, aerial_weight=0):
    return (ball_position[2] + common_values.BALL_RADIUS) / (2 * common_values.BALL_RADIUS) ** aerial_weight
