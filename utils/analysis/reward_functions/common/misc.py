from typing import Union, List, Tuple

import numpy as np
from rlgym.utils import common_values

from utils.math import cosine_similarity


def event(args: Union[Tuple[List[int]], Tuple[List[int], List[float]]],
          event_names=("goal", "team_goal", "concede", "touch", "shot", "save", "demo"),
          remove_events: Union[str, int, List[Union[int, str]]] = None,
          add_events: Union[str, List[str]] = None):
    """
    Event reward. Provides a sum of specified rewards
    :param args: A tuple of a list of event weights or a tuple of event flags and event weights.
        Event weights and flags must match event names.
    :param event_names: A list of event names
    :param remove_events: Name(s) or event index(ices) to remove from the default or a provided list of rewards
    :param add_events: Event names to append to the default or a provided list of rewards.
        Events are appended to the end of the list.
    """
    event_names = list(event_names)

    # Remove events from predefined events if needed
    if remove_events:
        if type(remove_events) != list:
            if type(remove_events) == int:
                del event_names[remove_events]
            else:
                event_names.remove(remove_events)
        else:
            for r_ev in remove_events:
                if type(r_ev) == int:
                    del event_names[r_ev]
                else:
                    event_names.remove(r_ev)

    # Add events to predefined events if needed
    if add_events:
        if type(add_events) != list:
            event_names.append(add_events)
        else:
            for a_ev in add_events:
                event_names.append(a_ev)

    # Event weights list case
    if len(args) == 1:
        event_weights = args[0]
        assert len(event_weights) == len(event_names), "Event weights do not match events"
        return np.sum(event_weights)
    # Event bools and weights case
    if len(args) == 2:
        events = args[0]
        weights = args[1]
        assert len(events) == len(event_names), "Event indicators do not match event names"
        assert len(weights) == len(events), "Event weights do not match event indicators"
        return np.dot(events, weights)

    raise Exception("Wrong number of arguments provided."
                    " Arguments can either be event weights or event indicators and weights.")


def velocity(player_lin_velocity, negative=False):
    return np.linalg.norm(player_lin_velocity, 2, axis=-1) / common_values.CAR_MAX_SPEED * (1 - 2 * negative)


def save_boost(boost_amount):
    return np.sqrt(boost_amount)


def constant(w):
    return w


def align_ball(player_position, ball_position, defense=1, offense=1, orange=False):
    if not orange:
        blue_goal = np.array(common_values.BLUE_GOAL_BACK)
        orange_goal = np.array(common_values.ORANGE_GOAL_BACK)
    else:
        blue_goal = np.array(common_values.ORANGE_GOAL_BACK)
        orange_goal = np.array(common_values.BLUE_GOAL_BACK)

    defensive = defense * cosine_similarity(ball_position - player_position, player_position - blue_goal)
    offensive = offense * cosine_similarity(ball_position - player_position, orange_goal - player_position)

    return defensive + offensive
