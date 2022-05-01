from rlgym.utils.reward_functions import common_rewards
from rlgym_tools.extra_rewards import kickoff_reward

from .ball_goal import BallYCoordinateReward, LiuDistanceBallToGoalReward, SignedLiuDistanceBallToGoalReward
from .extra import DiffPotentialReward
from .misc import DistanceWeightedAlignBallGoal, EventReward
from .player_ball import LiuDistancePlayerToBallReward, OffensivePotentialReward

rewards_names_map = {common_rewards.LiuDistanceBallToGoalReward: "Ball to goal distance",
                     common_rewards.VelocityBallToGoalReward: "Ball to goal velocity",
                     common_rewards.BallYCoordinateReward: "Ball y coordinate",
                     common_rewards.VelocityReward: "Player velocity",
                     common_rewards.SaveBoostReward: "Save boost",
                     common_rewards.ConstantReward: "Constant",
                     common_rewards.AlignBallGoal: "Align ball to goal",
                     common_rewards.LiuDistancePlayerToBallReward: "Player to ball distance",
                     common_rewards.VelocityPlayerToBallReward: "Player to ball velocity",
                     common_rewards.FaceBallReward: "Face ball",
                     common_rewards.TouchBallReward: "Touch ball aerial",
                     kickoff_reward.KickoffReward: "Kickoff",
                     BallYCoordinateReward: "Ball y coordinate",
                     LiuDistanceBallToGoalReward: "Ball to goal distance",
                     SignedLiuDistanceBallToGoalReward: "Signed ball to goal distance",
                     DistanceWeightedAlignBallGoal: "Distance-weighted align ball to goal",
                     OffensivePotentialReward: "Offensive potential",
                     LiuDistancePlayerToBallReward: "Player to ball distance",
                     }
