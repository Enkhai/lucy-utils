from typing import List, Union, Type, Sequence

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
    A function that returns an RLGym match
    """
    return Match(reward_function=reward,
                 terminal_conditions=terminal_conditions,
                 obs_builder=obs_builder,
                 state_setter=state_setter,
                 action_parser=action_parser,
                 team_size=team_size,
                 self_play=self_play,
                 game_speed=500)


def get_matches(reward: RewardFunction,
                terminal_conditions: Union[TerminalCondition, List[TerminalCondition]],
                obs_builder_cls: Type[ObsBuilder],
                action_parser_cls: Type[ActionParser] = KBMAction,
                state_setter_cls: Type[StateSetter] = DefaultState,
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
    # out of the three cls type arguments, observation builders should at least not be shared between matches
    # (class argument instead of object argument, initialization happens for each match)
    # that is because observation builders often maintain state data that is specific to each match
    return [get_match(reward,
                      terminal_conditions,
                      obs_builder_cls(),
                      action_parser_cls(),
                      state_setter_cls(),
                      size,
                      self_play)
            for size, self_play in zip(sizes, self_plays)]
