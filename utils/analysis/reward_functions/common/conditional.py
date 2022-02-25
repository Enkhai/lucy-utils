from typing import List, Union, Tuple

import numpy as np


def closest2ball_(player_positions, player_idx, team_idcs, ball_position, team_only=True):
    dist = np.linalg.norm(player_positions[player_idx] - ball_position, 2)
    for team_idx, player2_pos in zip(team_idcs, player_positions):
        if not team_only or team_idx == team_idcs[player_idx]:
            dist2 = np.linalg.norm(player2_pos - ball_position, 2)
            if dist2 < dist:
                return False
    return True


def behind_ball_(player_position, ball_position, orange=False):
    if orange:
        return player_position[1] > ball_position[1]
    return player_position[1] < ball_position[1]


def conditional(condition: str,
                condition_params: Union[bool, Tuple[Union[int, bool, List[int], np.ndarray]]] = True) -> bool:
    """
    Conditional reward
    :param condition: Available conditions are "closest2ball", "touched_last" and "behind_ball"
    :param condition_params: A boolean indicating the condition applies or a numpy array condition parameter
    :return: A floating point scalar
    """
    condition_map = {"closest2ball": closest2ball_,
                     "touched_last": None,
                     "behind_ball": behind_ball_}
    condition_function = condition_map[condition]
    # closest2ball, behind_ball
    if condition_function:
        if type(condition_params) is bool:
            return condition_params
        return condition_function(*condition_params)
    # touched_last
    elif condition in condition_map:
        return condition_params
    return False
