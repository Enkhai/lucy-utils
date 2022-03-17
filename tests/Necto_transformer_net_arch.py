from copy import copy

import numpy as np
from rlgym.utils.reward_functions import common_rewards, CombinedReward
from rlgym.utils.state_setters import DefaultState
from rlgym.utils.terminal_conditions import common_conditions
from rlgym_tools.extra_action_parsers.kbm_act import KBMAction
from rlgym_tools.extra_rewards.diff_reward import DiffReward
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor

from utils.algorithms import DeviceAlternatingPPO
from utils.multi_instance_utils import get_matches
from utils.obs import AttentionObs
from utils.policies import ACPerceiverPolicy

reward = CombinedReward.from_zipped(
    # reward shaping function
    (DiffReward(CombinedReward.from_zipped(
        (common_rewards.LiuDistanceBallToGoalReward(), 7),
        (common_rewards.VelocityBallToGoalReward(), 2),
        (common_rewards.BallYCoordinateReward(), 1),
        (common_rewards.VelocityPlayerToBallReward(), 0.5),
        (common_rewards.LiuDistancePlayerToBallReward(), 0.5),
        (common_rewards.AlignBallGoal(0.5, 0.5), 0.75)
    )), 1),
    # original reward
    (common_rewards.SaveBoostReward(), 0.4),
    (common_rewards.EventReward(goal=10, team_goal=4, concede=-10, touch=0.05, shot=1, save=3, demo=2), 1),
)
models_folder = "models/"

if __name__ == '__main__':
    # TODO: Fix n_steps (Necto: batch_size) and batch_size (Necto: mini_batch_size) computation,
    #  based on Impossibum's tutorial

    gamma = np.exp(np.log(0.5) / ((120 / 8) * 10))

    matches = get_matches(rewards=[copy(reward) for _ in range(8)],  # different reward for each match
                          terminal_conditions=[common_conditions.NoTouchTimeoutCondition(500),
                                               common_conditions.GoalScoredCondition()],
                          obs_builder_cls=AttentionObs,
                          state_setter_cls=DefaultState,
                          action_parser_cls=KBMAction,
                          sizes=[2] * 8  # 8-match 2v2 scenario
                          )
    env = SB3MultipleInstanceEnv(match_func_or_matches=matches, force_paging=True)
    env = VecMonitor(env)

    policy_kwargs = dict(net_arch=[dict(
        # minus one for the key padding mask
        query_dims=env.observation_space.shape[-1] - 1,
        # minus eight for the previous action
        kv_dims=env.observation_space.shape[-1] - 1 - 8,
        # the rest is default arguments
    )] * 2)  # *2 because actor and critic will share the same architecture

    # model = PPO.load("models/Perceiver/model_4096000_steps.zip", env, device=device)
    model = DeviceAlternatingPPO(policy=ACPerceiverPolicy,
                                 env=env,
                                 learning_rate=1e-4,
                                 n_steps=100_000,  # TODO: fix this
                                 gamma=gamma,
                                 batch_size=25_000,  # TODO: fix this
                                 tensorboard_log="./bin",
                                 policy_kwargs=policy_kwargs,
                                 verbose=1,
                                 )
    # TODO: add instantaneous fps callback
    callbacks = [CheckpointCallback(500_000,  # TODO: fix this
                                    save_path=models_folder + "Perceiver",
                                    name_prefix="model")]
    # 2 because separate actor and critic branches,
    # 4 because 4 perceiver blocks,
    # 256 because 256 perceiver block hidden dims
    model.learn(total_timesteps=1_000_000_000, callback=callbacks, tb_log_name="PPO_Perceiver2_4x256")
    model.save(models_folder + "Perceiver_final")

    env.close()
