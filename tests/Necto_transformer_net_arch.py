from rlgym.utils.reward_functions import common_rewards, CombinedReward
from rlgym.utils.state_setters import DefaultState
from rlgym.utils.terminal_conditions import common_conditions
from rlgym_tools.extra_action_parsers.kbm_act import KBMAction
from rlgym_tools.extra_rewards.diff_reward import DiffReward
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv
from rlgym_tools.sb3_utils.sb3_log_reward import SB3CombinedLogReward, SB3CombinedLogRewardCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor
from copy import deepcopy

from utils.algorithms import DeviceAlternatingPPO
from utils.multi_instance_utils import get_matches
# This example is dedicated to an attention-based network, inspired by an architecture named Perceiver
# `utils.obs` contains an observation builder appropriate for such a network, while `utils.policies` contains
# a policy that makes use of it
from utils.obs import AttentionObs
from utils.policies import ACPerceiverPolicy

reward = SB3CombinedLogReward.from_zipped(
    (DiffReward(common_rewards.LiuDistanceBallToGoalReward()), 7),
    (DiffReward(common_rewards.VelocityBallToGoalReward()), 2),
    (DiffReward(common_rewards.BallYCoordinateReward()), 1),
    (DiffReward(common_rewards.VelocityPlayerToBallReward()), 0.5),
    (DiffReward(common_rewards.LiuDistancePlayerToBallReward()), 0.5),
    (DiffReward(common_rewards.AlignBallGoal(0.5, 0.5)), 0.75),
    (common_rewards.EventReward(touch=0.05), 1),
    (common_rewards.SaveBoostReward(), 0.4),
    (common_rewards.EventReward(demo=2), 1),
    (common_rewards.EventReward(save=3), 1),
    (common_rewards.EventReward(goal=10, team_goal=4, concede=-10), 1),
)
reward_names = ["Ball2goal dist diff",
                "Ball2goal vel diff",
                "Ball y coord diff",
                "Player2ball vel diff",
                "Player2ball dist diff",
                "Align ball goal diff",
                "Touch",
                "Save boost",
                "Demo",
                "Save",
                "Goal"]
models_folder = "models/"

if __name__ == '__main__':
    matches = get_matches(rewards=[deepcopy(reward) for _ in range(6)],  # different reward for each match
                          terminal_conditions=[common_conditions.NoTouchTimeoutCondition(500),
                                               common_conditions.GoalScoredCondition()],
                          obs_builder_cls=AttentionObs,
                          state_setter_cls=DefaultState,
                          action_parser_cls=KBMAction,
                          sizes=[2] * 6  # 6-match 2v2 scenario
                          )
    env = SB3MultipleInstanceEnv(match_func_or_matches=matches)
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
                                 # Batch size dictates the minibatch size used for backpropagation
                                 # A larger batch size equals more general and faster model weight updates
                                 # The community makes use of batch sizes of ~25k and larger
                                 batch_size=1024,
                                 tensorboard_log="./bin",
                                 policy_kwargs=policy_kwargs,
                                 verbose=1,
                                 )
    callbacks = [SB3CombinedLogRewardCallback(reward_names),
                 CheckpointCallback(model.n_steps * 100,
                                    save_path=models_folder + "Perceiver",
                                    name_prefix="model")]
    # 2 because separate actor and critic branches,
    # 4 because 4 perceiver blocks,
    # 256 because 256 perceiver block hidden dims
    model.learn(total_timesteps=100_000_000, callback=callbacks, tb_log_name="PPO_Perceiver2_4x256")
    model.save(models_folder + "Perceiver_final")

    env.close()
