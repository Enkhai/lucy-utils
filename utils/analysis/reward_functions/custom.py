from typing import List, Tuple, Union

import numpy as np
from rlgym.utils import common_values

from utils.analysis.reward_functions import common

_goal_depth = common_values.BACK_NET_Y - common_values.BACK_WALL_Y + common_values.BALL_RADIUS


def signed_liu_dist_ball2goal(ball_position: np.ndarray, dispersion=1, density=1, own_goal=False):
    """
    A natural extension of a "Ball close to target" reward, inspired by https://arxiv.org/abs/2105.12196.\n
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
