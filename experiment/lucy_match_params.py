from pathlib import Path

from rlgym.utils.reward_functions import common_rewards
from rlgym.utils.state_setters import RandomState, DefaultState
from rlgym.utils.terminal_conditions import common_conditions
from rlgym_tools.extra_action_parsers.kbm_act import KBMAction
from rlgym_tools.extra_state_setters.goalie_state import GoaliePracticeState
from rlgym_tools.extra_state_setters.replay_setter import ReplaySetter
from rlgym_tools.extra_state_setters.symmetric_setter import KickoffLikeSetter
from rlgym_tools.extra_state_setters.weighted_sample_setter import WeightedSampleSetter

from utils import rewards
from utils.build_reward import build_logged_reward
from utils.obs import GraphAttentionObs

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


def _get_reward(gamma: float, log: bool = False):
    return build_logged_reward(_f_reward_weight_args, _r_reward_name_weight_args, 0.3, gamma, log)


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
LucyObs = GraphAttentionObs
LucyAction = KBMAction
