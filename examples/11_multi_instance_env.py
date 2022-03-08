from rlgym.envs.match import Match
from rlgym.utils.reward_functions import common_rewards
from rlgym.utils.state_setters import DefaultState
from rlgym.utils.terminal_conditions import common_conditions
from rlgym_tools.extra_action_parsers.kbm_act import KBMAction
from rlgym_tools.extra_rewards.diff_reward import DiffReward
from rlgym_tools.sb3_utils.sb3_log_reward import SB3CombinedLogReward, SB3CombinedLogRewardCallback
from rlgym_tools.sb3_utils.sb3_multiple_instance_env import SB3MultipleInstanceEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor

from utils.obs import SimpleObs

reward = SB3CombinedLogReward.from_zipped(
    (DiffReward(common_rewards.LiuDistancePlayerToBallReward()), 0.05),
    (DiffReward(common_rewards.LiuDistanceBallToGoalReward()), 10),
    (common_rewards.ConstantReward(), -0.004),
    (common_rewards.EventReward(touch=0.05, goal=10)),
)
reward_names = ["PlayerToBallDistDiff", "BallToGoalDistDiff", "ConstantNegative", "GoalOrTouch"]
models_folder = "models/"


# We need a method to create a match for each instance
# Unfortunately, we cannot have a method that accepts arguments to pass to, since RLGym does not support this
def get_match():
    return Match(reward_function=reward,
                 terminal_conditions=[common_conditions.TimeoutCondition(500),
                                      common_conditions.GoalScoredCondition()],
                 obs_builder=SimpleObs(),
                 state_setter=DefaultState(),
                 action_parser=KBMAction(),
                 game_speed=500)


if __name__ == '__main__':
    # Creating the environment may take some time, this is normal
    # Always make sure your computer can handle your multiple game instances
    # You can do this by checking your device RAM and making sure your pagefile size is large enough
    # To check and change the pagefile size consult the following
    # https://www.tomshardware.com/news/how-to-manage-virtual-memory-pagefile-windows-10,36929.html
    # Each instance takes about 3.5Gb space of RAM upon startup but only ~400Mb when minimized
    # Turning off unnecessary apps and services can also be useful
    env = SB3MultipleInstanceEnv(match_func_or_matches=get_match,
                                 num_instances=6,
                                 wait_time=20)
    # With multi-instance environments the mean reward and episode length are not be logged
    # To overcome this, wrap the environment with VecMonitor
    env = VecMonitor(env)

    policy_kwargs = dict(net_arch=[dict(vf=[256, 256, 256, 256],  # completely separate actor and critic architecture
                                        pi=[256, 256, 256, 256])
                                   ])
    model = PPO(policy="MlpPolicy",
                env=env,
                learning_rate=1e-4,
                tensorboard_log="./bin",
                policy_kwargs=policy_kwargs,
                verbose=1,
                device="cpu",
                )
    # Random seed doesn't work in multi-instance environments
    # model.set_random_seed(0)

    callbacks = [SB3CombinedLogRewardCallback(reward_names),
                 # The number of steps here are effectively multiplied by 6 (6 agent-controlled cars)
                 CheckpointCallback(model.n_steps * 100,
                                    save_path=models_folder + "MLP2",
                                    name_prefix="model")]
    model.learn(total_timesteps=100_000_000, callback=callbacks, tb_log_name="PPO_MLP2_4x256")
    model.save(models_folder + "MLP2_final")

    env.close()