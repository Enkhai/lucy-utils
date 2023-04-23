from pathlib import Path

import numpy as np
import torch as th

_path = str(Path(__file__).parent.resolve())


def _extract_features(features):
    """
    :param features: Composite torch tensor of NectoObs features
    :return: query, key-value, mask tensors
    """
    return (features[:, [0], :-1],
            features[:, 1:, :-9],
            features[:, 1:, -1])


class NextoActor:
    """
    Nexto model for model comparison
    """

    def __init__(self):
        super(NextoActor, self).__init__()
        self.nexto = th.jit.load(_path + '/models/nexto-model.pt')
        self._lookup_table = self.make_lookup_table()

    @staticmethod
    def make_lookup_table():
        actions = []
        # Ground
        for throttle in (-1, 0, 1):
            for steer in (-1, 0, 1):
                for boost in (0, 1):
                    for handbrake in (0, 1):
                        if boost == 1 and throttle != 1:
                            continue
                        actions.append([throttle or boost, steer, 0, steer, 0, 0, boost, handbrake])
        # Aerial
        for pitch in (-1, 0, 1):
            for yaw in (-1, 0, 1):
                for roll in (-1, 0, 1):
                    for jump in (0, 1):
                        for boost in (0, 1):
                            if jump == 1 and yaw != 0:  # Only need roll for sideflip
                                continue
                            if pitch == roll == jump == 0:  # Duplicate with ground
                                continue
                            # Enable handbrake for potential wavedashes
                            handbrake = jump == 1 and (pitch != 0 or yaw != 0 or roll != 0)
                            actions.append([boost, yaw, pitch, yaw, roll, jump, boost, handbrake])
        actions = np.array(actions)
        return actions

    def predict(self, state, deterministic=True):
        """
        Following SB3 algorithm predict() method.
        :param state: The last game state, as returned by NextoObsBuilder.
        :param deterministic: Deterministic, property not actually used.
        :return: Discrete 8-action numpy array output, None
        """
        q, kv, mask = _extract_features(th.from_numpy(state).float())
        out = self.nexto((q, kv, mask))[0]

        actions = self._lookup_table[out.argmax(-1)]

        return actions, None
