from typing import List, Union, Tuple

import numpy as np


def closest2ball_(player_positions, player_idx, team_idcs, ball_position, team_only=True):
    dist = np.linalg.norm(player_positions[player_idx] - ball_position)
    for team_idx, player2_pos in zip(team_idcs, player_positions):
        if not team_only or team_idx == team_idcs[player_idx]:
            dist2 = np.linalg.norm(player2_pos - ball_position)
            if dist2 < dist:
                return False
    return True


def behind_ball_(player_position, ball_position):
    return player_position[1] > ball_position[1]


def conditional(condition: str,
                condition_params: Tuple[Union[int, bool, List[int], np.ndarray]] = True) -> bool:
    """
    Conditional reward
    :param condition: Available conditions are "closest2ball", "touched_last" and "behind_ball"
    :param condition_params: A boolean indicating the condition applies or a numpy array condition parameter
    :return: A floating point scalar
    """
    cond_map = {"closest2ball": closest2ball_,
                "touched_last": None,
                "behind_ball": behind_ball_}
    cond = cond_map[condition]
    if cond:
        return cond(*condition_params)
    elif condition in cond_map:
        return True
    return False
