from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv
from rlgym_tools.sb3_utils.sb3_instantaneous_fps_callback import SB3InstantaneousFPSCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor

from experiment.lucy_match_params import LucyReward, LucyTerminalConditions, LucyObs, LucyState, LucyAction
from utils.algorithms import DeviceAlternatingPPO
from utils.multi_instance_utils import get_matches, config, get_match
from utils.policies import ACPerceiverPolicy
from utils.rewards.sb3_log_reward import SB3NamedLogRewardCallback

models_folder = "models/"

if __name__ == '__main__':
    num_instances = 8
    agents_per_match = 2 * 2  # self-play
    n_steps, batch_size, gamma, fps, save_freq = config(num_instances=num_instances,
                                                        avg_agents_per_match=agents_per_match,
                                                        target_steps=256_0,
                                                        target_batch_size=0.5,
                                                        callback_save_freq=10)

    logger_match = get_match(reward=LucyReward(log=True),
                             terminal_conditions=LucyTerminalConditions(fps),
                             obs_builder=LucyObs(),
                             action_parser=LucyAction(),
                             state_setter=LucyState(),
                             team_size=agents_per_match // 2,  # self-play, hence // 2
                             self_play=True)

    matches = get_matches(reward_cls=LucyReward,
                          # minus the logger match
                          terminal_conditions=[LucyTerminalConditions(fps) for _ in range(num_instances - 1)],
                          obs_builder_cls=LucyObs,
                          state_setter_cls=LucyState,
                          action_parser_cls=LucyAction,
                          self_plays=True,
                          # self-play, hence // 2
                          sizes=[agents_per_match // 2] * (num_instances - 1)  # minus the logger match
                          )
    matches = [logger_match] + matches

    env = SB3MultipleInstanceEnv(match_func_or_matches=matches)
    env = VecMonitor(env)

    policy_kwargs = dict(net_arch=[dict(
        # minus one for the key padding mask
        query_dims=env.observation_space.shape[-1] - 1,
        # minus eight for the previous action
        kv_dims=env.observation_space.shape[-1] - 1 - 8,
        # the rest is default arguments
    )] * 2)  # *2 because actor and critic will share the same architecture

    # model = DeviceAlternatingPPO.load("../models/Perceiver/model_449280000_steps.zip", env)
    model = DeviceAlternatingPPO(policy=ACPerceiverPolicy,
                                 env=env,
                                 learning_rate=1e-4,
                                 n_steps=n_steps,
                                 gamma=gamma,
                                 batch_size=batch_size,
                                 tensorboard_log="./bin",
                                 policy_kwargs=policy_kwargs,
                                 verbose=1,
                                 )

    callbacks = [SB3InstantaneousFPSCallback(),
                 SB3NamedLogRewardCallback(),
                 CheckpointCallback(save_freq,
                                    save_path=models_folder + "Perceiver",
                                    name_prefix="model")]
    model.learn(total_timesteps=1_000_000_000,
                callback=callbacks,
                # 2 because separate actor and critic branches,
                # 4 because 4 perceiver blocks,
                # 256 because 256 perceiver block hidden dims
                tb_log_name="PPO_Perceiver2_4x256",
                reset_num_timesteps=False)
    model.save(models_folder + "Perceiver_final")

    env.close()
