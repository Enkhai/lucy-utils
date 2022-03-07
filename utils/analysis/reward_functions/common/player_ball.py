import numpy as np
from rlgym.utils import common_values, math


def liu_dist_player2ball(player_position, ball_position):
    # Inspired by https://arxiv.org/abs/2105.12196
    dist = np.linalg.norm(player_position - ball_position, 2, axis=-1) - common_values.BALL_RADIUS
    return np.exp(-0.5 * dist / common_values.CAR_MAX_SPEED)


def velocity_player2ball(player_position,
                         player_lin_velocity,
                         ball_position,
                         use_scalar_projection=False):
    pos_diff = ball_position - player_position
    if use_scalar_projection:
        return math.scalar_projection(player_lin_velocity, pos_diff)
    else:
        norm_pos_dif = pos_diff / (np.linalg.norm(pos_diff, 2, axis=-1)[:, None] + 1e-8)
        player_lin_velocity = player_lin_velocity / common_values.CAR_MAX_SPEED
        return np.dot(norm_pos_dif, player_lin_velocity)


def face_ball(player_position, ball_position, player_forward_vec):
    pos_diff = ball_position - player_position
    norm_pos_diff = pos_diff / (np.linalg.norm(pos_diff, 2, axis=-1)[:, None] + 1e-8)
    return (norm_pos_diff * player_forward_vec).sum(-1)


def touch_ball(ball_position, aerial_weight=0):
    return ((ball_position[2] + common_values.BALL_RADIUS) / (2 * common_values.BALL_RADIUS)) ** aerial_weight
