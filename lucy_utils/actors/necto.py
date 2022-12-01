from pathlib import Path

import torch as th
from torch.nn import functional as F

_path = str(Path(__file__).parent.resolve())


def _extract_features(features):
    """
    :param features: Composite torch tensor of NectoObs features
    :return: query, key-value, mask tensors
    """
    return (features[:, [0], :-1],
            features[:, 1:, :-9],
            features[:, 1:, -1])


class NectoActor:
    """
    Necto models for model comparison. Versions include:
     - regular: RLBot Pack Necto model, default
     - 10Y: Necto model trained for 10 years (~4.25 billion time steps)
     - 20Y: Necto model trained for 20 years (~9.46 billion time steps)
     - 30Y: Necto model trained for 30 years (~14.19 billion time steps)
    """

    models = {"regular": _path + "/models/necto-model.pt",
              "10Y": _path + "/models/necto-model-10Y.pt",
              "20Y": _path + "/models/necto-model-20Y.pt",
              "30Y": _path + "/models/necto-model-30Y.pt"}

    def __init__(self, version: str = "regular"):
        super(NectoActor, self).__init__()
        assert version in self.models.keys()
        self.necto = th.jit.load(self.models[version])

    def predict(self, state, deterministic=False):
        """
        Following SB3 algorithm predict() method
        :param state: The last game state, as returned by NectoObs
        :param deterministic: Deterministic, property not actually used.
        :return: KBM 5-action numpy array output, None
        """
        q, kv, mask = _extract_features(th.from_numpy(state).float())
        out = self.necto((q, kv, mask))[0]

        max_shape = max(o.shape[-1] for o in out)
        logits = th.stack(
            [
                l
                if l.shape[-1] == max_shape
                else F.pad(l, pad=(0, max_shape - l.shape[-1]), value=float("-inf"))
                for l in out
            ]
        ).swapdims(0, 1).squeeze()

        return logits.argmax(-1).numpy(), None
