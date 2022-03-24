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
from utils.rewards.sb3_log_reward import SB3NamedLogReward


def _get_reward(log: bool = False):
    """
    Reward for regular and logger matches.
    Set `logger` to None for regular matches, set to a custom logger for logger matches.
    """

    # reward shaping function
    f = DiffReward(CombinedReward.from_zipped(
        (SB3NamedLogReward(rewards.SignedLiuDistanceBallToGoalReward(),
                           "Signed distance ball from goal", "utility", log=log), 8),
        (SB3NamedLogReward(common_rewards.VelocityBallToGoalReward(),
                           "Velocity ball to goal", "utility", log=log), 2),
        (SB3NamedLogReward(rewards.BallYCoordinateReward(),
                           "Ball y coordinate", "utility", log=log), 1),
        (SB3NamedLogReward(common_rewards.VelocityPlayerToBallReward(),
                           "Velocity player to ball", "utility", log=log), 0.5),
        (SB3NamedLogReward(rewards.LiuDistancePlayerToBallReward(),
                           "Distance player to ball", "utility", log=log), 0.5),
        (SB3NamedLogReward(rewards.DistanceWeightedAlignBallGoal(0.5, 0.5),
                           "Distance-weighted align ball to goal", "utility", log=log), 0.65),
        (SB3NamedLogReward(common_rewards.SaveBoostReward(),
                           "Save boost", "utility", log=log), 0.5)))
    f = SB3NamedLogReward(f, "Reward shaping function", log=log)

    # original reward
    r = CombinedReward.from_zipped(
        (SB3NamedLogReward(rewards.EventReward(goal=10, team_goal=4, concede=-10), "Goal", log=log), 1),
        (SB3NamedLogReward(rewards.EventReward(shot=1), "Shot", log=log), 1),
        (SB3NamedLogReward(rewards.EventReward(save=3), "Save", log=log), 1),
        (SB3NamedLogReward(rewards.EventReward(touch=0.05), "Touch", log=log), 1),
        (SB3NamedLogReward(rewards.EventReward(demo=2, demoed=-2), "Demo", log=log), 1)
    )
    r = SB3NamedLogReward(r, "Original reward", log=log)

    total_reward = SB3NamedLogReward(CombinedReward.from_zipped((f, 1), (r, 1)), "Reward total", log=log)
    distributed_total_reward = SB3NamedLogReward(DistributeRewards(total_reward), "Distributed reward total", log=log)

    return distributed_total_reward


def _get_terminal_conditions(fps):
    return [common_conditions.TimeoutCondition(fps * 300),
            common_conditions.NoTouchTimeoutCondition(fps * 45),
            common_conditions.GoalScoredCondition()]


def _get_state():
    # Following Necto logic
    return WeightedSampleSetter.from_zipped(
        # replay setter uses carball, no warnings for numpy==1.21.5
        (ReplaySetter("../replay-samples/2v2/states.npy"), 0.7),
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
