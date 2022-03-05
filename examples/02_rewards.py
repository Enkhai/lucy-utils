import rlgym
from rlgym.utils.reward_functions import CombinedReward
from rlgym.utils.reward_functions.common_rewards import FaceBallReward, \
    TouchBallReward, \
    LiuDistanceBallToGoalReward, \
    LiuDistancePlayerToBallReward, \
    VelocityPlayerToBallReward, \
    VelocityBallToGoalReward
from rlgym.utils.state_setters import DefaultState
from rlgym.utils.terminal_conditions.common_conditions import GoalScoredCondition, TimeoutCondition
from rlgym_tools.extra_obs.advanced_stacker import AdvancedStacker
from stable_baselines3 import TD3

# For training purposes only, it can be useful to set graphics settings in Rocket League to minimum,
# set a low windowed resolution, and change frame rate to uncapped

if __name__ == '__main__':  # Required for multi-instance training

    # Use the following reward functions
    fns = (LiuDistanceBallToGoalReward(),
           VelocityBallToGoalReward(),
           TouchBallReward(),
           VelocityPlayerToBallReward(),
           LiuDistancePlayerToBallReward(),
           FaceBallReward())

    # with this importance\weighting
    fn_weights = (5, 4, 3, 2, 2, 1)

    # to produce a complex combined reward
    # *This is not a very good reward function...
    combined_reward = CombinedReward(fns, fn_weights)

    # RLGym environments are at their core gym.core.Gym objects and can be used with many kinds of different
    # reinforcement learning libraries
    # For this case, our agent will only attempt to learn to shoot a ball in the opponent team's goal
    env = rlgym.make(game_speed=500,
                     tick_skip=8,
                     spawn_opponents=False,
                     self_play=False,
                     team_size=1,
                     terminal_conditions=[TimeoutCondition(500), GoalScoredCondition()],
                     # DefaultReward is for demonstration purposes only and does nothing useful
                     reward_fn=combined_reward,
                     obs_builder=AdvancedStacker(),  # A more advanced observation builder
                     state_setter=DefaultState(),  # This resets all cars back to kickoff position
                     use_injector=False  # Requires True for multiple game instances
                     )
    model = TD3("MlpPolicy", env, verbose=1)

    model.learn(total_timesteps=1_000_000)

    env.close()
    # For useful rlgym examples you can also check out rlgym_tools.examples
