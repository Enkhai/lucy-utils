from typing import List, Any, Tuple

import gym.spaces
import numpy as np
from rlgym.utils.action_parsers import ActionParser
from rlgym.utils.gamestates import GameState


class MixedAction(ActionParser):

    def __init__(self,
                 parsers: List[ActionParser],
                 num_parser_players: List[int]):
        super(MixedAction, self).__init__()
        self.parsers = parsers
        offsets = np.concatenate([0], np.cumsum(num_parser_players))
        self.parser_idx = [offsets[i:i + 2] for i in range(len(num_parser_players))]

    def get_action_space(self) -> gym.spaces.Space:
        return gym.spaces.Tuple((p.get_action_space() for p in self.parsers))

    def parse_actions(self, actions: Any, state: GameState) -> Tuple[np.ndarray]:
        return tuple(p.parse_actions(actions[n[0]:n[1]], state)
                     for p, n in zip(self.parsers, self.parser_idx))