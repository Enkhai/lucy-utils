from typing import Sequence, Tuple, Type, Union, Optional

from rlgym.utils import RewardFunction
from rlgym.utils.reward_functions import CombinedReward
from rlgym_tools.extra_rewards.distribute_rewards import DistributeRewards

from .rewards import rewards_names_map
from .rewards.extra import DiffPotentialReward
from .rewards.sb3_log_reward import SB3NamedLogReward


def build_logged_reward(f_rews: Sequence[Tuple[Type[RewardFunction], Union[int, float], Optional[dict]]],
                        r_rews: Sequence[Tuple[Type[RewardFunction], str, Union[int, float], dict]],
                        team_spirit: float = 0.3,
                        gamma: float = 0.99,
                        log: bool = False) -> SB3NamedLogReward:
    """
    Builds a complete composite potential-based shaped reward function, with a team spirit factor, allowing for logging
    of individual reward components. Suitable for regular and logger matches. Works in conjunction with
    the `SB3NamedLogRewardCallback`.

    The reward follows the potential-based reward shaping logic with the complete shaped reward R' defined as:\n
    R' = R + F

    where:\n
    R = r_rews[0][0](**r_rews[0][3]) * r_rews[0][2] + r_rews[1][0](**r_rews[1][3]) * r_rews[1][2]` + ... +
    r_rews[n][0](**r_rews[n][3]) * r_rews[n][2]

    and\n
    F = γΦ(s') - Φ(s), with\n
    Φ = f_rews[0][0](**f_rews[0][2]) * f_rews[0][1] + f_rews[1][0](**f_rews[1][2]) * f_rews[1][1]` + ... +
    f_rews[n][0](**f_rews[n][2]) * f_rews[n][1]

    Additionally, R' is distributed among team players, using a team spirit factor as such:\n
    R'_i = avg(R'_team) * team_spirit + R'_i * (1 - team_spirit) - avg(R'_opponent)

    :param f_rews: A sequence of potential rewards passed as tuples of shape
     (reward_class, reward_weight (, kwargs))
    :param r_rews: A sequence of event rewards passed as tuples of shape
     (reward_class, reward_name, reward_weight, kwargs).
    :param team_spirit: Team spirit factor for reward distribution
    :param gamma: Gamma value used for the reward shaping function
    :param log: Dictates whether to log individual reward components. Set to True for logger matches.
    """

    # reward shaping function
    f_zip = ()
    for f_rew in f_rews:
        try:
            f_rew_args = f_rew[2]
        except IndexError:
            f_rew_args = {}
        f_zip += ((SB3NamedLogReward(f_rew[0](**f_rew_args), rewards_names_map[f_rew[0]],
                                     "utility", log=log), f_rew[1]),)
    f = SB3NamedLogReward(DiffPotentialReward(CombinedReward.from_zipped(*f_zip), gamma),
                          "Reward shaping function", log=log)

    # original reward
    r = CombinedReward.from_zipped(*((SB3NamedLogReward(r_rew[0](**r_rew[3]), r_rew[1], log=log), r_rew[2])
                                     for r_rew in r_rews))
    r = SB3NamedLogReward(r, "Original reward", log=log)

    total_reward = SB3NamedLogReward(CombinedReward.from_zipped((f, 1), (r, 1)), "Reward total", log=log)
    distributed_total_reward = SB3NamedLogReward(DistributeRewards(total_reward, team_spirit),
                                                 "Distributed reward total", log=log)

    return distributed_total_reward
