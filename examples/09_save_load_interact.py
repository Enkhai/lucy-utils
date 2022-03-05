import rlgym
from rlgym.utils.reward_functions import common_rewards
from rlgym.utils.terminal_conditions.common_conditions import GoalScoredCondition, TimeoutCondition
from rlgym_tools.sb3_utils.sb3_log_reward import SB3CombinedLogReward, SB3CombinedLogRewardCallback
from stable_baselines3 import PPO


# Here we give an example on how to train and save a model and how to furtherly load it
# and have it play on a regular environment
# We create a method that will train a simple PPO model and save it at the end,
# similarly to what we have already done so far
def train_and_save():
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
                     terminal_conditions=[TimeoutCondition(500), GoalScoredCondition()],
                     reward_fn=reward)

    policy_kwargs = dict(net_arch=[128, 128, dict(vf=[64, 32], pi=[64, 32, 16])])
    model = PPO(policy="MlpPolicy",
                env=env,
                tensorboard_log="./bin",
                verbose=1,
                policy_kwargs=policy_kwargs,
                device="cpu")
    model.set_random_seed(0)

    reward_log_callback = SB3CombinedLogRewardCallback(reward_names=reward_names)
    model.learn(total_timesteps=1_000_000, callback=reward_log_callback)

    # This is the only difference
    # Algorithms that also make use of experience replay, such as the off-policy HER algorithm
    # can also save the replay buffer to continue training further on
    model.save("model")

    env.close()


# A simple method that will load our model and have it play for a maximum of 100 episodes
def load_and_play():
    # We create our environment anew
    # We don't need to specify an elaborate reward function, since we are only interacting with the game
    # and not learning
    env = rlgym.make(game_speed=1,
                     terminal_conditions=[TimeoutCondition(500), GoalScoredCondition()])

    # You must always make sure that the environment you are building uses the same observation
    # builder the model was trained on
    # If you are using a different observation builder than the default, say AdvancedObs,
    # you should declare it in the `make` arguments
    # env = rlgym.make(game_speed=1,
    #                  terminal_conditions=[TimeoutCondition(500), GoalScoredCondition()],
    #                  obs_builder=AdvancedObs(),
    #                  self_play=True)

    # Also, if you are using multiple agents, make sure you wrap your environment with the according environment
    # wrapper
    # env = SB3SingleInstanceEnv(env)

    # We load our model
    model = PPO.load(path="model", env=env, device="cpu")

    # And repeat for 100 episodes
    for _ in range(100):
        obs = env.reset()
        # For multiple agents use lists for obs, reward, done and gameinfo variables
        # done = [False]
        done = False

        # The done variable should have the same value for all agents at each step
        # while not done[0]:
        while not done:
            # action = model.predict(obs)[0]
            action = model.predict(obs)
            obs, reward, done, gameinfo = env.step(action)

    env.close()


if __name__ == '__main__':
    train_and_save()
    load_and_play()
