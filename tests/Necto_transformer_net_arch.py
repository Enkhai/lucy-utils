import os

from rlgym.utils.reward_functions import common_rewards, CombinedReward
from rlgym.utils.state_setters import DefaultState
from rlgym.utils.state_setters import RandomState
from rlgym.utils.terminal_conditions import common_conditions
from rlgym_tools.extra_action_parsers.kbm_act import KBMAction
from rlgym_tools.extra_rewards.diff_reward import DiffReward
from rlgym_tools.extra_rewards.distribute_rewards import DistributeRewards
from rlgym_tools.extra_state_setters.goalie_state import GoaliePracticeState
from rlgym_tools.extra_state_setters.replay_setter import ReplaySetter
from rlgym_tools.extra_state_setters.symmetric_setter import KickoffLikeSetter
from rlgym_tools.extra_state_setters.weighted_sample_setter import WeightedSampleSetter
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv
from rlgym_tools.sb3_utils.sb3_instantaneous_fps_callback import SB3InstantaneousFPSCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import configure_logger
from stable_baselines3.common.vec_env import VecMonitor

from utils import rewards
from utils.algorithms import DeviceAlternatingPPO
from utils.multi_instance_utils import get_matches, config
from utils.obs import AttentionObs
from utils.rewards.sb3_log_reward import SB3NamedBlueLogReward

# TODO: add logger to SB3LogRewards
logger = configure_logger(verbose=1,
                          tensorboard_log="./bin",
                          tb_log_name="PPO_Perceiver2_4x256",
                          reset_num_timesteps=False)


# TODO: change to separate SB3LogRewards
def get_reward():
    return DistributeRewards(CombinedReward.from_zipped(
        # reward shaping function
        (DiffReward(CombinedReward.from_zipped(
            (rewards.SignedLiuDistanceBallToGoalReward(), 8),
            (common_rewards.VelocityBallToGoalReward(), 2),
            (common_rewards.BallYCoordinateReward(), 1),
            (common_rewards.VelocityPlayerToBallReward(), 0.5),
            (common_rewards.LiuDistancePlayerToBallReward(), 0.5),
            (rewards.DistanceWeightedAlignBallGoal(0.5, 0.5), 0.65),
            (common_rewards.SaveBoostReward(), 0.5)
        )), 1),
        # original reward
        (common_rewards.EventReward(goal=10, team_goal=4, concede=-10, touch=0.05, shot=1, save=3, demo=2), 1),
    ))


def get_log_reward():
    # TODO: fill this with SB3NamedBlueLogReward
    pass


def get_state():
    # Following Necto logic
    replay_folder = "replay-samples/2v2/"
    return WeightedSampleSetter.from_zipped(
        (ReplaySetter.construct_from_replays(list(replay_folder + f for f in os.listdir(replay_folder))), 0.7),
        (RandomState(True, True, False), 0.15),
        (DefaultState(), 0.05),
        (KickoffLikeSetter(), 0.05),
        (GoaliePracticeState(first_defender_in_goal=True), 0.05)
    )


models_folder = "models/"

if __name__ == '__main__':
    num_instances = 8
    agents_per_match = 2 * 2  # self-play
    n_steps, batch_size, gamma, fps, save_freq = config(num_instances=num_instances,
                                                        avg_agents_per_match=agents_per_match,
                                                        target_steps=256_000,
                                                        target_batch_size=0.5,
                                                        callback_save_freq=10)

    matches = get_matches(reward_cls=get_reward,
                          terminal_conditions=[[common_conditions.TimeoutCondition(fps * 300),
                                                common_conditions.NoTouchTimeoutCondition(fps * 45),
                                                common_conditions.GoalScoredCondition()]
                                               for _ in range(num_instances)],
                          obs_builder_cls=AttentionObs,
                          state_setter_cls=get_state,
                          action_parser_cls=KBMAction,
                          self_plays=True,
                          # self-play, hence // 2
                          sizes=[agents_per_match // 2] * num_instances
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

    model = DeviceAlternatingPPO.load("models/Perceiver/model_308480000_steps.zip", env)
    # model = DeviceAlternatingPPO(policy=ACPerceiverPolicy,
    #                              env=env,
    #                              learning_rate=1e-4,
    #                              n_steps=n_steps,
    #                              gamma=gamma,
    #                              batch_size=batch_size,
    #                              tensorboard_log="./bin",
    #                              policy_kwargs=policy_kwargs,
    #                              verbose=1,
    #                              )
    model.set_logger(logger)

    callbacks = [SB3InstantaneousFPSCallback(),
                 CheckpointCallback(save_freq,
                                    save_path=models_folder + "Perceiver",
                                    name_prefix="model")]
    # 2 because separate actor and critic branches,
    # 4 because 4 perceiver blocks,
    # 256 because 256 perceiver block hidden dims
    model.learn(total_timesteps=1_000_000_000,
                callback=callbacks,
                tb_log_name="PPO_Perceiver2_4x256",  # this is pointless when setting a custom logger
                reset_num_timesteps=False)
    model.save(models_folder + "Perceiver_final")

    env.close()
