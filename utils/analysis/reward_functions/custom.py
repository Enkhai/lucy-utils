from typing import List, Tuple, Union

import numpy as np
from rlgym.utils import common_values

from utils.analysis.reward_functions import common

_goal_depth = common_values.BACK_NET_Y - common_values.BACK_WALL_Y + common_values.BALL_RADIUS


def offensive_potential(player_position,
                        ball_position,
                        player_lin_velocity,
                        defense=0.5,
                        offense=0.5,
                        dispersion=1,
                        density=1,
                        orange=False):
    """
    Offensive potential function. When the player to ball and ball to goal vectors align
    we should reward player to ball velocity.\n
    Uses a combination of `AlignBallGoal`,`VelocityPlayerToBallReward` and `LiuDistancePlayerToBallReward` rewards.
    """

    velocity_player2ball_rew = common.velocity_player2ball(player_position, player_lin_velocity, ball_position)
    align_ball_rew = common.align_ball(player_position, ball_position,
                                       defense, offense, orange)
    liu_dist_player2ball_rew = liu_dist_player2ball(player_position, ball_position, dispersion, density)

    # logical AND
    # when both alignment and player to ball velocity are negative we must get a negative output
    sign = (((velocity_player2ball_rew >= 0) & (align_ball_rew >= 0)) - 0.5) * 2
    # liu_dist_player2ball_rew is positive only, no need to compute for sign
    rew = align_ball_rew * velocity_player2ball_rew * liu_dist_player2ball_rew
    # cube root because we multiply three values between -1 and 1
    # "weighted" product (n_1 * n_2 * ... * n_N) ^ (1 / N)
    return (np.abs(rew) ** (1 / 3)) * sign


def dist_weighted_align_ball(player_position,
                             ball_position,
                             defense=0.5,
                             offense=0.5,
                             dispersion=1,
                             density=1,
                             orange=False):
    align_ball_rew = common.align_ball(player_position, ball_position,
                                       defense, offense, orange)
    liu_dist_player2ball_rew = liu_dist_player2ball(player_position, ball_position, dispersion, density)

    rew = align_ball_rew * liu_dist_player2ball_rew
    # square root because we multiply two values between -1 and 1
    # "weighted" product (n_1 * n_2 * ... * n_N) ^ (1 / N)
    return np.sqrt(np.abs(rew)) * np.sign(rew)


def signed_liu_dist_ball2goal(ball_position: np.ndarray, dispersion=1, density=1, own_goal=False):
    """
    A natural extension of a signed "Ball close to target" reward, inspired by https://arxiv.org/abs/2105.12196.\n
    Produces an approximate reward of 0 at ball position [side_wall, 0, ball_radius].
    """
    objective = np.array(common_values.ORANGE_GOAL_BACK) if not own_goal \
        else np.array(common_values.BLUE_GOAL_BACK)

    # Distance is computed with respect to the goal back adjusted by the goal depth
    dist = np.linalg.norm(ball_position - objective, 2, axis=-1) - _goal_depth

    # with dispersion
    # trigonometry solution - produces an approximate unsigned value of 0.5 at position [4096, 0, 93]
    rew = np.exp(-0.5 * dist / (4570 * dispersion))
    # signed
    rew = (rew - 0.5) * 2
    # with density
    rew = (np.abs(rew) ** (1 / density)) * np.sign(rew)

    return rew


def liu_dist_ball2goal(ball_position: np.ndarray, dispersion=1, density=1, own_goal=False):
    """
    A natural extension of a "Ball close to target" reward, inspired by https://arxiv.org/abs/2105.12196.
    """
    objective = np.array(common_values.ORANGE_GOAL_BACK) if not own_goal \
        else np.array(common_values.BLUE_GOAL_BACK)

    # Distance is computed with respect to the goal back adjusted by the goal depth
    dist = np.linalg.norm(ball_position - objective, 2, axis=-1) - _goal_depth

    # with dispersion
    rew = np.exp(-0.5 * dist / (common_values.BALL_MAX_SPEED * dispersion))
    # with density
    rew = rew ** (1 / density)

    return rew


def liu_dist_player2ball(player_position, ball_position, dispersion=1, density=1):
    """
    A natural extension of a "Player close to ball" reward, inspired by https://arxiv.org/abs/2105.12196
    """
    dist = np.linalg.norm(player_position - ball_position, 2, axis=-1) - common_values.BALL_RADIUS
    return np.exp(-0.5 * dist / (common_values.CAR_MAX_SPEED * dispersion)) ** (1 / density)


def diff_potential(reward, gamma, negative_slope=1):
    """
    Potential-based reward shaping function with a `negative_slope` magnitude parameter
    """
    return ((gamma * reward[1:]) - reward[:-1]) * negative_slope


def ball_y_coord(ball_position, exponent=1):
    rew = ball_position[:, 1] / (common_values.BACK_WALL_Y + common_values.BALL_RADIUS)
    rew = (np.abs(rew) ** exponent) * np.sign(rew)
    return rew


def event(args: Union[Tuple[List[int]], Tuple[List[int], List[float]]],
          event_names=("goal", "team_goal", "concede", "touch", "shot", "save", "demo", "demoed"),
          remove_events: Union[str, int, List[Union[int, str]]] = None,
          add_events: Union[str, List[str]] = None):
    """
    Custom event reward with additional `demoed` reward pre-specified. Provides a sum of specified rewards.
    :param args: A tuple of a list of event weights or a tuple of event flags and event weights.
        Event weights and flags must match event names.
    :param event_names: A list of event names
    :param remove_events: Name(s) or event index(ices) to remove from the default or a provided list of rewards
    :param add_events: Event names to append to the default or a provided list of rewards.
        Events are appended to the end of the list.
    """
    return common.event(args, event_names, remove_events, add_events)
