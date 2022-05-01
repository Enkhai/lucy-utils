from typing import List, Union, Type, Sequence, Callable, Tuple
from typing_extensions import Protocol

import numpy as np
from rlgym.envs import Match
from rlgym.utils.action_parsers import ActionParser
from rlgym.utils.obs_builders import ObsBuilder
from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.state_setters import DefaultState
from rlgym.utils.state_setters import StateSetter
from rlgym.utils.terminal_conditions import TerminalCondition
from rlgym_tools.extra_action_parsers.kbm_act import KBMAction

from .rewards.sb3_log_reward import SB3NamedLogReward


class LoggedRewardBuilder(Protocol):
    """
    Callable signature for logged reward function builders
    """

    def __call__(self, log: bool = False) -> SB3NamedLogReward: ...


def get_match(reward: RewardFunction,
              terminal_conditions: Union[TerminalCondition, List[TerminalCondition]],
              obs_builder: ObsBuilder,
              action_parser: ActionParser,
              state_setter: StateSetter,
              team_size: int,
              self_play=True):
    """
    A function that returns an RLGym match
    """
    return Match(reward_function=reward,
                 terminal_conditions=terminal_conditions,
                 obs_builder=obs_builder,
                 action_parser=action_parser,
                 state_setter=state_setter,
                 team_size=team_size,
                 self_play=self_play,
                 game_speed=500)


def get_matches(reward_cls: Union[Type[RewardFunction], Callable[[], RewardFunction]],
                terminal_conditions: Callable[[], Union[TerminalCondition, Sequence[TerminalCondition]]],
                obs_builder_cls: Union[Type[ObsBuilder], Callable[[], ObsBuilder]],
                action_parser_cls: Union[Type[ActionParser], Callable[[], ActionParser]] = KBMAction,
                state_setter_cls: Union[Type[StateSetter], Callable[[], StateSetter]] = DefaultState,
                self_plays: Union[bool, Sequence[bool]] = True,
                sizes: List[int] = None):
    """
    A function useful for creating a number of matches for multi-instance environments.\n
    If sizes is None or empty a list of `[3, 3, 2, 2, 1, 1]` sizes is used instead.
    """
    if not sizes:
        sizes = [3, 3, 2, 2, 1, 1]
    if type(self_plays) == bool:
        self_plays = [self_plays] * len(sizes)
    assert len(sizes) == len(self_plays), "Size mismatch between `sizes` and `self_plays`"

    return [get_match(reward_cls(),
                      terminal_conditions(),
                      obs_builder_cls(),
                      action_parser_cls(),
                      state_setter_cls(),
                      size,
                      self_play)
            for size, self_play in zip(sizes, self_plays)]


def make_matches(logged_reward_cls: LoggedRewardBuilder,
                 terminal_conditions: Callable[[], Union[TerminalCondition, Sequence[TerminalCondition]]],
                 obs_builder_cls: Union[Type[ObsBuilder], Callable[[], ObsBuilder]],
                 action_parser_cls: Union[Type[ActionParser], Callable[[], ActionParser]] = KBMAction,
                 state_setter_cls: Union[Type[StateSetter], Callable[[], StateSetter]] = DefaultState,
                 self_plays: Union[bool, Sequence[bool]] = True,
                 sizes: List[int] = None,
                 add_logger_match: bool = True) -> Sequence[Match]:
    """
    Creates a list of matches, suitable for multi-instance environments. Similar to `get_matches`.

    Requires a reward builder function with a `log=False` boolean parameter that returns an `SB3NamedLogReward` for
    logging reward component values.

    If `add_logger_match=True` the first match returned in the list is a logger match that has reward logging enabled.
    Rewards logged from the logger match can be written to Tensorboard using an `SB3NamedLogRewardCallback`.
    """
    if not sizes:
        sizes = [3, 3, 2, 2, 1, 1]
    if type(self_plays) == bool:
        self_plays = [self_plays] * len(sizes)
    assert len(sizes) == len(self_plays), "Size mismatch between `sizes` and `self_plays`"

    matches = []
    if add_logger_match:
        matches += [get_match(reward=logged_reward_cls(log=True),
                              terminal_conditions=terminal_conditions(),
                              obs_builder=obs_builder_cls(),
                              action_parser=action_parser_cls(),
                              state_setter=state_setter_cls(),
                              team_size=sizes[0],
                              self_play=self_plays[0])]

    if len(sizes) > add_logger_match:
        matches += get_matches(reward_cls=logged_reward_cls,
                               terminal_conditions=terminal_conditions,
                               obs_builder_cls=obs_builder_cls,
                               action_parser_cls=action_parser_cls,
                               state_setter_cls=state_setter_cls,
                               self_plays=self_plays[add_logger_match:],
                               sizes=sizes[add_logger_match:])
    return matches


def config(num_instances: int,
           avg_agents_per_match: int,
           target_steps: int,
           half_life_seconds: Union[float, int] = 10.0,
           target_batch_size: Union[float, int] = 0.1,
           callback_save_freq: int = 5,
           frame_skip: int = 8) -> Tuple[int, int, float, int, int]:
    """
    A configuration function that computes training hyperparameters necessary for multi-instance environments.\n
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

    fps = 120 / frame_skip
    assert fps % 1 == 0, "120 physics frames per second not integer divisible by `frame_skip`."
    fps = int(fps)

    gamma = np.exp(np.log(0.5) / (fps * half_life_seconds))

    n_steps = target_steps / (num_instances * avg_agents_per_match)
    assert n_steps % 1 == 0, "Number of steps cannot be equally distributed between players." \
                             "Please check your `target_steps`, `num_instances` and `avg_agents_per_match` parameters."
    n_steps = int(n_steps)

    if target_batch_size <= 1:
        batch_size = n_steps * target_batch_size
        assert batch_size % 1 == 0, "Batch size is not an integer number. " \
                                    "Please check your `target_batch_size` parameter."
    else:
        batch_size = target_batch_size / (num_instances * avg_agents_per_match)
        assert batch_size % 1 == 0, "Batch size is not an integer number. Please check your `target_batch_size`, " \
                                    "`num_instances` and `avg_agents_per_match` parameters."
    batch_size = int(batch_size)

    save_freq = int(n_steps * callback_save_freq)
    return n_steps, batch_size, gamma, fps, save_freq
