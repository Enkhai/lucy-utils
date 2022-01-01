from typing import Any

import numpy as np
from rlgym.utils import ObsBuilder, common_values
from rlgym.utils.gamestates import GameState, PlayerData


class SimpleObs(ObsBuilder):
    """
    Simple observation builder for a ball and one car only
    Observation space is of shape 1 * 2 * 20:
    1 (batch)
    * 2 (1 ball + 1 car)
    * 20 (2 (car and ball flags)
        + 9 ((relative) standardized position, linear velocity and angular velocity  3-d vectors)
        + 6 (forward and upward rotation axes 3-d vectors)
        + 3 (boost, touching ground and has flip flags))

    If flatten is true, it simply returns a vector of length 40 (2 * 20)
    """
    POS_STD = 3000

    def __init__(self, flatten: bool = False):
        super(SimpleObs, self).__init__()
        self.flatten = flatten

    def reset(self, initial_state: GameState):
        pass

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:
        ball = state.ball
        car = player.car_data

        ball_obs = np.concatenate([[1, 0],
                                   (ball.position - car.position) / self.POS_STD,
                                   (ball.linear_velocity - car.linear_velocity) / common_values.BALL_MAX_SPEED,
                                   ball.angular_velocity / common_values.CAR_MAX_ANG_VEL,
                                   np.zeros(6),
                                   np.zeros(3)])
        car_obs = np.concatenate([[0, 1],
                                  car.position / self.POS_STD,
                                  car.linear_velocity / common_values.CAR_MAX_SPEED,
                                  car.angular_velocity / common_values.CAR_MAX_ANG_VEL,
                                  car.forward(),
                                  car.up(),
                                  [player.boost_amount, player.on_ground, player.has_flip]])
        if self.flatten:
            return np.concatenate([ball_obs, car_obs])
        return np.stack([ball_obs, car_obs])
