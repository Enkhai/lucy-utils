import rlgym
from rlgym.utils.reward_functions import common_rewards
from rlgym.utils.terminal_conditions.common_conditions import GoalScoredCondition, TimeoutCondition
from rlgym_tools.sb3_utils.sb3_log_reward import SB3CombinedLogReward, SB3CombinedLogRewardCallback
from stable_baselines3 import PPO
from rlgym_tools.sb3_utils import SB3SingleInstanceEnv

if __name__ == '__main__':
    reward = SB3CombinedLogReward.from_zipped(
        (common_rewards.ConstantReward(), -0.02),
        (common_rewards.EventReward(goal=1, concede=-1), 100),
        (common_rewards.VelocityPlayerToBallReward(), 0.05),
        (common_rewards.VelocityBallToGoalReward(), 0.2),
        (common_rewards.TouchBallReward(), 0.2),
        (common_rewards.VelocityReward(), 0.01),
        (common_rewards.LiuDistanceBallToGoalReward(), 0.25),
        (common_rewards.LiuDistancePlayerToBallReward(), 0.1),
        (common_rewards.AlignBallGoal(), 0.15),
        (common_rewards.FaceBallReward(), 0.1)
    )
    reward_names = [fn.__class__.__name__ for fn in reward.reward_functions]

    env = rlgym.make(game_speed=500,
                     # spawn_opponents spawns All-star Psyonix bots to play against with as opponents
                     # If the game speed is very high, however, the bots become much worse (possibly due to packet loss)
                     # spawn_opponents is generally not recommended, since there is a bug
                     # that allows all other cars besides the main player's to be controlled by bots, including
                     # teammates'
                     # spawn_opponents=True,
                     # By setting self_play to True enables a model to control both team sides
                     # Having multiple cars to control, as is the case for self_play, transforms the environment
                     # to require a list of actions for each player
                     # and return a list of observations, rewards, done flags and game info, in a `gym` fashion
                     # The reward function and state space can also be built differently for each agent individually,
                     # separating logic between agents
                     self_play=True,
                     terminal_conditions=[TimeoutCondition(500), GoalScoredCondition()],
                     reward_fn=reward)
    # To enable self-play, we use the SB3SingleInstanceEnv wrapper
    env = SB3SingleInstanceEnv(env)

    model = PPO("MlpPolicy",
                env,
                policy_kwargs={"net_arch": [{
                    "pi": [32, 16, 8, 4],
                    "vf": [32, 16, 8, 4]
                }]},
                tensorboard_log="./bin",
                verbose=1,
                device="cpu")
    # By setting a random seed to the model we feed this seed to
    # the `random` python module, numpy, torch, the OpenAI `gym` module used by RLGym, and the action space
    model.set_random_seed(0)

    reward_log_callback = SB3CombinedLogRewardCallback(reward_names=reward_names)
    model.learn(total_timesteps=100_000_000, callback=reward_log_callback)

    env.close()
