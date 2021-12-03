# Draft - Will be used for running experiments
import rlgym
from rlgym.utils.reward_functions import CombinedReward
from rlgym.utils.reward_functions.common_rewards import *
from rlgym_tools.extra_rewards.diff_reward import DiffReward
from rlgym.utils.terminal_conditions.common_conditions import GoalScoredCondition, TimeoutCondition
from rlgym_tools.extra_obs.advanced_stacker import AdvancedStacker
from stable_baselines3 import PPO

from utils import *


# Will be used for constructing a CombinedReward based on the experiment
def make_reward(df):
    pass


# Will be used for building the model based on the experiment
def make_model(df):
    pass


# Will be used for initializing the environment based on the experiment
def make_world(df):
    pass


if __name__ == '__main__':
    # Load the experiment
    exp_id = int(input("Experiment id:"))

    experiment_df = load_hyperparams(exp_id)

    # Make the reward
    reward = make_reward(experiment_df)

    # Make the model
    model = make_model(experiment_df)

    # Initialize the environment
    env = make_world(experiment_df)

    # Train for some time steps
    # Callbacks can be added for saving the model at certain points
    # Tensorboard can be introduced
    model.learn(total_timesteps=100_000_000)

    # Save the model
    model.save('model.zip')

    # Close the environment and quit Rocket League
    env.close()
