from typing import List, Any

import gym.spaces
import numpy as np
from rlgym.utils.action_parsers import ActionParser, DiscreteAction
from rlgym.utils.gamestates import GameState

from .actors import NextoActor


class MixedAction(ActionParser):

    def __init__(self,
                 parsers: List[ActionParser],
                 num_parser_players: List[int]):
        super(MixedAction, self).__init__()
        self.parsers = parsers
        offsets = np.concatenate([[0], np.cumsum(num_parser_players)])
        self.parser_idx = [offsets[i:i + 2] for i in range(len(num_parser_players))]
        self.action_lengths = [p.get_action_space().shape[0] for p in parsers]

    def get_action_space(self) -> gym.spaces.Space:
        return gym.spaces.Box(-np.inf, np.inf, (8,))

    def parse_actions(self, actions: Any, state: GameState) -> np.ndarray:
        return np.concatenate([p.parse_actions(actions[n[0]:n[1], :l], state)
                               for p, l, n in zip(self.parsers, self.action_lengths, self.parser_idx)])


class NextoAction(ActionParser):
    def __init__(self):
        self._lookup_table = NextoActor.make_lookup_table()
        self._discrete_parser = DiscreteAction()
        super().__init__()

    def get_action_space(self) -> gym.spaces.Space:
        return gym.spaces.Discrete(90)

    def parse_actions(self, actions: Any, state: GameState) -> np.ndarray:
        return self._discrete_parser.parse_actions(self._lookup_table[actions], state)
