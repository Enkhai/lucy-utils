import rlgym
from rlgym.utils.reward_functions.common_rewards import *
from rlgym.utils.terminal_conditions.common_conditions import GoalScoredCondition, TimeoutCondition
from rlgym_tools.extra_obs.advanced_stacker import AdvancedStacker
from rlgym_tools.sb3_utils.sb3_log_reward import SB3CombinedLogReward  # Useful for separate combined reward logging
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter
from multiprocessing.queues import SimpleQueue


class RewardLogCallback(BaseCallback):

    def __init__(self, log_dir: str,
                 reward_names: list,
                 read_queue: SimpleQueue,
                 comment: str = "",
                 verbose=0):
        super(RewardLogCallback, self).__init__(verbose)
        # Init stuff
        # A thread could be useful that reads the file as it is written and
        # sends new line data into a queue that will be read by the Callback
        # TODO: Make the thread
        self.read_queue = read_queue
        self.writer = SummaryWriter(log_dir=log_dir, comment=comment)
        # The reward class names will be read during the combined reward initialization
        # and passed to the callback to add to scalars
        self.reward_names = reward_names
        # Having a list initialized from the start can be faster than constantly
        # creating a new one at each step
        self.reward_values = []

    def _on_step(self) -> bool:
        # Here we will be reading the log txt file to log in Tensorboard

        self.reward_values.clear()
        # Parse the log file from the queue
        while not self.read_queue.empty():
            self.reward_values.append(self.read_queue.get())

        # Sort the rewards and compute an average - TODO

        # Log the rewards as scalars - TODO
        # example
        self.writer.add_scalar(tag=self.reward_names[0],
                               scalar_value=0,
                               global_step=self.num_timesteps,
                               new_style=True)  # what is new_style?

        # _on_step must return a boolean
        # If the boolean is false training is aborted early
        return True


if __name__ == '__main__':
    # Logs the combined rewards for each episode in a txt file inside a ./bin folder
    # This is useful for studying what the model learns to do best and how it may exploit the rewards
    # to obtain a maximum return
    # TODO: Create a list of reward class names for the callback
    reward = SB3CombinedLogReward.from_zipped(
        (ConstantReward(), -0.02),
        (EventReward(goal=1, concede=-1), 100),
        (VelocityPlayerToBallReward(), 0.05),
        (VelocityBallToGoalReward(), 0.2),
        (TouchBallReward(), 0.2),
        (VelocityReward(), 0.01),
        (LiuDistanceBallToGoalReward(), 0.25),
        (LiuDistancePlayerToBallReward(), 0.1),
        (AlignBallGoal(), 0.15),
        (FaceBallReward(), 0.1)
    )

    # The observation space depends on the observation builder
    # By default, the observation builder is the DefaultObs, which returns an unbounded observation space of size 70
    # We can also use other observation builders such as the AdvancedStacker,
    # which returns an observation space of size 196
    env = rlgym.make(game_speed=500,
                     terminal_conditions=[TimeoutCondition(500), GoalScoredCondition()],
                     reward_fn=reward)  # , obs_builder=AdvancedStacker())

    # The CnnPolicy can only be applied to an image observation space
    # In Rocket League, the observation space is a vector that depends on the observation builder
    # Depending on the size of the observation space, we can build our model accordingly
    # For MLP model building rules-of-thumb read this:
    # https://towardsdatascience.com/17-rules-of-thumb-for-building-a-neural-network-93356f9930af
    # The action space is a vector of size 8, bounded by -1 and 1, based on the rule that
    # most reinforcement learning algorithms rely on a Gaussian distribution,
    # initially centered around 0 with std 1, for continuous actions
    # A consequence of Gaussian distributions, however, is that if the action space
    # is unbounded and not normalized between -1 and 1, this can harm learning and be difficult to debug
    # You can read more on this on the Stable Baselines 3 guide:
    # https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html
    # The action space consists of the following actions:
    # backwards - S, forward - W,
    # steer left - A, steer right - D,
    # flip left - Q, flip right - E,
    # jump - right click, boost - left click
    # (what about power slide (Left Shift)?)
    model = PPO("MlpPolicy",
                env,
                policy_kwargs={"net_arch": [{
                    "pi": [32, 16, 8, 4],
                    "vf": [32, 16, 8, 4]
                }]},
                tensorboard_log="./bin",
                verbose=1,
                device="cpu")

    # TODO: continue studying to find a way to log the different combined rewards
    #  from the log file in ./bin in Tensorboard

    # You can observe the model's performance in Tensorboard by running in a terminal
    # `tensorboard --logdir bin`
    # You can then open up your browser and go to localhost:6006 to study the performance
    # If you want to have multiple Tensorboard servers running at the same time you can declare
    # a different port by running
    # `tensorboard --logdir <some_folder> --port <some other port, eg. 6007>`
    # TODO: implement the callback for logging episode rewards
    model.learn(total_timesteps=100_000_000)

    env.close()
