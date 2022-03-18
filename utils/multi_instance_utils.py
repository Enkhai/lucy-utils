from typing import List, Union, Type, Sequence, Callable, Tuple

import numpy as np
from rlgym.envs import Match
from rlgym.utils.action_parsers import ActionParser
from rlgym.utils.obs_builders import ObsBuilder
from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.state_setters import DefaultState
from rlgym.utils.state_setters import StateSetter
from rlgym.utils.terminal_conditions import TerminalCondition
from rlgym_tools.extra_action_parsers.kbm_act import KBMAction


def get_match(reward: RewardFunction,
              terminal_conditions: Union[TerminalCondition, List[TerminalCondition]],
              obs_builder: ObsBuilder,
              action_parser: ActionParser,
              state_setter: StateSetter,
              team_size: int,
              self_play=True):
    """
    A method that returns an RLGym match
    """
    return Match(reward_function=reward,
                 terminal_conditions=terminal_conditions,
                 obs_builder=obs_builder,
                 state_setter=state_setter,
                 action_parser=action_parser,
                 team_size=team_size,
                 self_play=self_play,
                 game_speed=500)


def get_matches(reward_cls: Union[Type[RewardFunction], Callable],
                terminal_conditions: Sequence[Union[TerminalCondition, Sequence[TerminalCondition]]],
                obs_builder_cls: Type[ObsBuilder],
                action_parser_cls: Type[ActionParser] = KBMAction,
                state_setter_cls: Type[StateSetter] = DefaultState,
                self_plays: Union[bool, Sequence[bool]] = True,
                sizes: List[int] = None):
    """
    A method useful for creating a number of matches for multi-instance environments.\n
    If sizes is None or empty a list of `[3, 3, 2, 2, 1, 1]` sizes is used instead.
    """
    if not sizes:
        sizes = [3, 3, 2, 2, 1, 1]
    if type(self_plays) == bool:
        self_plays = [self_plays] * len(sizes)
    # Class type arguments should not be shared between matches
    # (class argument instead of object argument, initialization happens for each match)
    # Must maintain state data that is specific to each match
    return [get_match(reward_cls(),
                      terminal_cond,
                      obs_builder_cls(),
                      action_parser_cls(),
                      state_setter_cls(),
                      size,
                      self_play)
            for terminal_cond, size, self_play in zip(terminal_conditions, sizes, self_plays)]


def config(num_instances: int,
           avg_agents_per_match: int,
           target_steps: int,
           half_life_seconds: Union[float, int] = 10.0,
           target_batch_size: Union[float, int] = 0.1,
           callback_save_freq: int = 5,
           frame_skip: int = 8) -> Tuple[int, int, float, int, int]:
    """
    A configuration method that computes training hyperparameters necessary for multi-instance environments.\n
    `target_batch_size` can either be a percentage or an actual size.

    :param num_instances: Number of environment instances, int
    :param avg_agents_per_match: Average number of environment instances, int
    :param target_steps: Target number of total rollout steps, int
    :param half_life_seconds: Number of seconds it takes for the gamma exponential to reach 0.5, int or float
    :param target_batch_size: Target batch size, int
    :param callback_save_freq: `CheckpointCallback` save frequency in terms of number of steps, int
    :param frame_skip: Number of frames to skip during training
    :return: Number of steps, batch size, gamma, fps and callback save model step frequency
    """
    assert target_batch_size > 0

    fps = 120 // frame_skip
    gamma = np.exp(np.log(0.5) / (fps * half_life_seconds))
    n_steps = target_steps // (num_instances * avg_agents_per_match)
    if target_batch_size <= 1:
        batch_size = int(n_steps * target_batch_size)
    else:
        batch_size = int(target_batch_size / (num_instances * avg_agents_per_match))
    save_freq = int(n_steps * callback_save_freq)
    return n_steps, batch_size, gamma, fps, save_freq
