import rlgym
from rlgym.utils.reward_functions import CombinedReward
from rlgym.utils.reward_functions.common_rewards import EventReward,\
    VelocityPlayerToBallReward, \
    VelocityBallToGoalReward, \
    TouchBallReward, \
    SaveBoostReward, \
    VelocityReward
from rlgym_tools.extra_rewards.diff_reward import DiffReward
from rlgym.utils.terminal_conditions.common_conditions import GoalScoredCondition, TimeoutCondition
from rlgym_tools.extra_obs.advanced_stacker import AdvancedStacker
from stable_baselines3 import PPO

if __name__ == '__main__':

    # Taken from the RLGym Discord server
    # A more complex network (more MLP layers, CNN layers) might be useful
    combined_reward = CombinedReward.from_zipped(
        (EventReward(goal=1, concede=-1), 5),
        (VelocityPlayerToBallReward(), 0.05),
        (VelocityBallToGoalReward(), 0.2),
        (TouchBallReward(), 0.02),
        (SaveBoostReward(), 0.05),
        (VelocityReward(), 0.06),
        (DiffReward(VelocityReward()), 0.06)
    )

    env = rlgym.make(game_speed=500,
                     terminal_conditions=[TimeoutCondition(500), GoalScoredCondition()],
                     reward_fn=combined_reward,
                     obs_builder=AdvancedStacker())

    model = PPO("MlpPolicy", env, verbose=1)

    model.learn(total_timesteps=1_000_000)

    env.close()
