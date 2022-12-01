from typing import List, Any

import numpy as np
from rlgym.utils import ObsBuilder
from rlgym.utils.gamestates import GameState, PlayerData


class MultiModelObs(ObsBuilder):

    def __init__(self, obss: List[ObsBuilder], num_obs_players: List[int]):
        super(MultiModelObs, self).__init__()
        assert len(obss) == len(num_obs_players), "'obss' and 'num_obs_players' lengths must match"
        self.obss = obss
        self.num_obs_players = np.cumsum(num_obs_players)
        self.p_idx = 0
        self.curr_state = None
        self.autodetect = True

    def reset(self, initial_state: GameState):
        [o.reset(initial_state) for o in self.obss]

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:
        if self.autodetect:
            self.autodetect = False
            return np.zeros(0)

        if self.curr_state != state:
            self.p_idx = 0
            self.curr_state = state

        obs_idx = (self.p_idx >= self.num_obs_players).sum()
        self.p_idx += 1

        return self.obss[obs_idx].build_obs(player, state, previous_action)
