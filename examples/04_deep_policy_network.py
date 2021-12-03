import rlgym
from rlgym.utils.reward_functions import CombinedReward
from rlgym.utils.reward_functions.common_rewards import *
from rlgym_tools.extra_rewards.diff_reward import DiffReward
from rlgym.utils.terminal_conditions.common_conditions import GoalScoredCondition, TimeoutCondition
from rlgym_tools.extra_obs.advanced_stacker import AdvancedStacker
from stable_baselines3 import PPO

if __name__ == '__main__':
    # combined_reward = CombinedReward.from_zipped(
    #     (EventReward(goal=1, concede=-1), 5),
    #     (VelocityPlayerToBallReward(), 0.03),
    #     (VelocityBallToGoalReward(), 0.2),
    #     (TouchBallReward(), 0.08),
    #     (VelocityReward(), 0.02),
    #     (DiffReward(VelocityReward()), 0.02),
    #     (LiuDistanceBallToGoalReward(), 0.2)
    # )

    combined_reward = CombinedReward.from_zipped(
        (ConstantReward(), -0.02),  # small negative reward for each tick
        (EventReward(goal=1, concede=-1), 100),  # large goal reward
        (VelocityPlayerToBallReward(), 0.05),  # the player should move towards to the ball
        (VelocityBallToGoalReward(), 0.2),  # the ball should certainly move towards the opponent's goal
        (TouchBallReward(), 0.2),  # touching the ball is good
        (VelocityReward(), 0.01),  # small velocity reward
        (LiuDistanceBallToGoalReward(), 0.25),  # the ball should certainly be moving towards the goal
        (LiuDistancePlayerToBallReward(), 0.1),  # the player should be moving towards the ball
        (AlignBallGoal(), 0.15),  # having the ball align towards the opponents goal is good
        (FaceBallReward(), 0.1)  # facing the ball might be good
    )

    env = rlgym.make(game_speed=500,
                     terminal_conditions=[TimeoutCondition(500), GoalScoredCondition()],
                     reward_fn=combined_reward,
                     obs_builder=AdvancedStacker())

    model = PPO("MlpPolicy",
                env,
                # To check which policy keyworded arguments can be used study the ActorCriticPolicy arguments
                # It's the same as the MlpPolicy for PPO, used for building the MLP policy network
                # For CNN network policy keyworded arguments check ActorCriticCnnPolicy
                policy_kwargs={"net_arch": [{
                    "pi": [64, 128, 128, 64],
                    "vf": [64, 128, 128, 64]
                }]},
                verbose=1)

    model.learn(total_timesteps=100_000_000)

    env.close()
