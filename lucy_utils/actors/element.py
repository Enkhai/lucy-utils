import pickle
from pathlib import Path

import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as F

_path = str(Path(__file__).parent.resolve())


class ElementActor:
    """
    Element actor for model comparison.
    """

    def __init__(self, num_players=2):
        super(ElementActor, self).__init__()
        model = pickle.load(open(_path + "/models/element.p", "rb"))
        self.element = _ElementModel()
        self.element.load_state_dict(model)
        self.num_players = num_players

    def predict(self, state, deterministic=True):
        """
        Following SB3 algorithm predict() method.
        :param state: The last game state, as returned by AdvancedObs.
        :param deterministic: Deterministic, property not actually used.
        :return: DiscreteAction 8-action numpy array output, None
        """
        state = np.stack(state)
        # get closest enemies
        enemies = state[:, (76 + (self.num_players - 1) * 31):].reshape((-1, self.num_players, 31))
        idcs = np.argmin(np.linalg.norm(enemies[:, :, 25:28], axis=-1), -1)
        closest = np.stack([enemies[i, c] for i, c in enumerate(idcs)])
        # update the state with closest only
        state = th.tensor(np.concatenate([state[:, :76], closest], -1), dtype=th.float)
        out = self.element(state)
        return th.cat([th.argmax(out[0], -1), th.argmax(out[1], -1)], -1).numpy(), None


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
