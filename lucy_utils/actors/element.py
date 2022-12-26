import pickle

import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as F

from . import _path


class ElementActor:
    """
    Element actor for model comparison.
    """
    _action_trans = np.array([-1] * 5 + [0] * 3)

    def __init__(self):
        super(ElementActor, self).__init__()
        model = pickle.load(open(_path + "/element.p", "rb"))
        self.element = _ElementModel()
        self.element.load_state_dict(model)

    def predict(self, state, deterministic=True):
        """
        Following SB3 algorithm predict() method.
        :param state: The last game state, as returned by AdvancedObs.
        :param deterministic: Deterministic, property not actually used.
        :return: DiscreteAction 8-action numpy array output, None
        """
        state = th.tensor(state, dtype=th.float)
        out = self.element(state)
        return th.cat([th.argmax(out[0], -1), th.argmax(out[1], -1)]).numpy() - self._action_trans


class _ElementModel(nn.Module):
    """
    Element model architecture.
    """

    def __init__(self):
        super(_ElementModel, self).__init__()
        self.fc1 = nn.Linear(107, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 256)
        self.cat_heads = nn.Linear(256, 3 * 5)
        self.ber_heads = nn.Linear(256, 2 * 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        cat_output = F.softmax(self.cat_heads(x).view(-1, 5, 3), dim=2)
        ber_output = F.softmax(self.ber_heads(x).view(-1, 3, 2), dim=2)
        return [cat_output, ber_output]
