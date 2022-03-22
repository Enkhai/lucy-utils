from rlgym.utils.reward_functions import CombinedReward, common_rewards
from rlgym.utils.state_setters import DefaultState
from rlgym.utils.terminal_conditions import common_conditions
from rlgym_tools.extra_action_parsers.kbm_act import KBMAction
from rlgym_tools.extra_rewards.diff_reward import DiffReward
from rlgym_tools.extra_rewards.distribute_rewards import DistributeRewards

from utils import rewards
from utils.load_save import load_and_evaluate
from utils.obs import AttentionObs

reward_fn = DistributeRewards(CombinedReward.from_zipped(
    # reward shaping function
    (DiffReward(CombinedReward.from_zipped(
        (rewards.SignedLiuDistanceBallToGoalReward(), 8),
        (common_rewards.VelocityBallToGoalReward(), 2),
        (common_rewards.BallYCoordinateReward(), 1),
        (common_rewards.VelocityPlayerToBallReward(), 0.5),
        (common_rewards.LiuDistancePlayerToBallReward(), 0.5),
        (rewards.DistanceWeightedAlignBallGoal(0.5, 0.5), 0.65),
        (common_rewards.SaveBoostReward(), 0.5)
    )), 1),
    # original reward
    (common_rewards.EventReward(goal=10, team_goal=4, concede=-10, touch=0.05, shot=1, save=3, demo=2), 1),
))

if __name__ == '__main__':
    fps = 120 // 8

    load_and_evaluate("../models/model_311040000_steps.zip",
                      2,
                      [common_conditions.TimeoutCondition(fps * 300),
                       common_conditions.NoTouchTimeoutCondition(fps * 45),
                       common_conditions.GoalScoredCondition()],
                      AttentionObs(),
                      DefaultState(),
                      KBMAction(),
                      reward_fn
                      )
