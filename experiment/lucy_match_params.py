from pathlib import Path
from typing import Type, Union, Optional, Tuple, Sequence

from rlgym.utils import RewardFunction
from rlgym.utils.reward_functions import CombinedReward, common_rewards
from rlgym.utils.state_setters import RandomState, DefaultState
from rlgym.utils.terminal_conditions import common_conditions
from rlgym_tools.extra_action_parsers.kbm_act import KBMAction
from rlgym_tools.extra_rewards.diff_reward import DiffReward
from rlgym_tools.extra_rewards.distribute_rewards import DistributeRewards
from rlgym_tools.extra_state_setters.goalie_state import GoaliePracticeState
from rlgym_tools.extra_state_setters.replay_setter import ReplaySetter
from rlgym_tools.extra_state_setters.symmetric_setter import KickoffLikeSetter
from rlgym_tools.extra_state_setters.weighted_sample_setter import WeightedSampleSetter

from utils import rewards
from utils.obs import AttentionObs
from utils.rewards import rewards_names_map
from utils.rewards.sb3_log_reward import SB3NamedLogReward

# potential: reward, weight (, args)
_f_reward_weight_args = ((rewards.SignedLiuDistanceBallToGoalReward, 8),
                         (common_rewards.VelocityBallToGoalReward, 2),
                         (rewards.BallYCoordinateReward, 1),
                         (common_rewards.VelocityPlayerToBallReward, 0.5),
                         (rewards.LiuDistancePlayerToBallReward, 0.5),
                         (rewards.DistanceWeightedAlignBallGoal, 0.65, dict(defense=0.5, offense=0.5)),
                         (common_rewards.SaveBoostReward, 0.5))
"""
Potential: reward class, weight (, kwargs)
"""

_r_reward_name_weight_args = ((rewards.EventReward, "Goal", 1, dict(goal=10, team_goal=4, concede=-10)),
                              (rewards.EventReward, "Shot", 1, dict(shot=1)),
                              (rewards.EventReward, "Save", 1, dict(save=3)),
                              (rewards.EventReward, "Touch", 1, dict(touch=0.05)),
                              (rewards.EventReward, "Demo", 1, dict(demo=2, demoed=-2)))
"""
Event: reward class, reward name, weight, kwargs
"""


def _get_reward(
        f_rews: Sequence[Tuple[Type[RewardFunction], Union[int, float], Optional[dict]]] = _f_reward_weight_args,
        r_rews: Sequence[Tuple[Type[RewardFunction], str, Union[int, float], dict]] = _r_reward_name_weight_args,
        log: bool = False):
    """
    Reward for regular and logger matches. Set `log=True` for logger matches.

    :param f_rews: A sequence of potential rewards passed as tuples of shape
     (reward_class, reward_weight (, kwargs))
    :param r_rews: A sequence of event rewards passed as tuples of shape
     (reward_class, reward_name, reward_weight, kwargs).
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
    f = SB3NamedLogReward(DiffReward(CombinedReward.from_zipped(*f_zip)), "Reward shaping function", log=log)

    # original reward
    r = CombinedReward.from_zipped(*((SB3NamedLogReward(r_rew[0](**r_rew[3]), r_rew[1], log=log), r_rew[2])
                                     for r_rew in r_rews))
    r = SB3NamedLogReward(r, "Original reward", log=log)

    total_reward = SB3NamedLogReward(CombinedReward.from_zipped((f, 1), (r, 1)), "Reward total", log=log)
    distributed_total_reward = SB3NamedLogReward(DistributeRewards(total_reward), "Distributed reward total", log=log)

    return distributed_total_reward


def _get_terminal_conditions(fps):
    return [common_conditions.TimeoutCondition(fps * 300),
            common_conditions.NoTouchTimeoutCondition(fps * 45),
            common_conditions.GoalScoredCondition()]


def _get_state():
    replay_path = str(Path(__file__).parent / "../replay-samples/2v2/states.npy")
    # Following Necto logic
    return WeightedSampleSetter.from_zipped(
        # replay setter uses carball, no warnings for numpy==1.21.5
        (ReplaySetter(replay_path), 0.7),
        (RandomState(True, True, False), 0.15),
        (DefaultState(), 0.05),
        (KickoffLikeSetter(), 0.05),
        (GoaliePracticeState(first_defender_in_goal=True), 0.05)
    )


LucyReward = _get_reward
LucyTerminalConditions = _get_terminal_conditions
LucyState = _get_state
LucyObs = AttentionObs
LucyAction = KBMAction
