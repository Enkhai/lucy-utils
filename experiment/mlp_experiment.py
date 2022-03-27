from rlgym.utils.action_parsers import DiscreteAction
from rlgym.utils.obs_builders import AdvancedObs
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv
from rlgym_tools.sb3_utils.sb3_instantaneous_fps_callback import SB3InstantaneousFPSCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor

from experiment.lucy_match_params import LucyReward, LucyTerminalConditions, LucyState
from utils.algorithms import DeviceAlternatingPPO
from utils.multi_instance_utils import get_matches, config, get_match
from utils.rewards.sb3_log_reward import SB3NamedLogRewardCallback

models_folder = "models/"

if __name__ == '__main__':
    num_instances = 8
    agents_per_match = 2 * 2  # self-play
    n_steps, batch_size, gamma, fps, save_freq = config(num_instances=num_instances,
                                                        avg_agents_per_match=agents_per_match,
                                                        target_steps=256_000,
                                                        target_batch_size=0.5,
                                                        callback_save_freq=10)

    logger_match = get_match(reward=LucyReward(log=True),
                             terminal_conditions=LucyTerminalConditions(fps),
                             obs_builder=AdvancedObs(),
                             action_parser=DiscreteAction(),
                             state_setter=LucyState(),
                             team_size=agents_per_match // 2,  # self-play, hence // 2
                             self_play=True)

    matches = get_matches(reward_cls=LucyReward,
                          # minus the logger match
                          terminal_conditions=[LucyTerminalConditions(fps) for _ in range(num_instances - 1)],
                          obs_builder_cls=AdvancedObs,
                          state_setter_cls=LucyState,
                          action_parser_cls=DiscreteAction,
                          self_plays=True,
                          # self-play, hence // 2
                          sizes=[agents_per_match // 2] * (num_instances - 1)  # minus the logger match
                          )
    matches = [logger_match] + matches

    env = SB3MultipleInstanceEnv(match_func_or_matches=matches)
    env = VecMonitor(env)

    # policy_kwargs = dict(
    #     net_arch=[512, 512, dict(pi=[256, 256, 256], vf=[256, 256, 256])],
    # )
    model = DeviceAlternatingPPO.load("./models/MLP_1_2x512_2_3x256/model_248320000_steps.zip", env)
    from stable_baselines3.common.utils import constant_fn

    model.clip_range = constant_fn(0.3)
    # model = DeviceAlternatingPPO(policy="MlpPolicy",
    #                              env=env,
    #                              learning_rate=1e-4,
    #                              n_steps=n_steps,
    #                              batch_size=batch_size,
    #                              gamma=gamma,
    #                              ent_coef=0.01,
    #                              vf_coef=1,
    #                              tensorboard_log="./bin",
    #                              policy_kwargs=policy_kwargs,
    #                              verbose=1)

    callbacks = [SB3InstantaneousFPSCallback(),
                 SB3NamedLogRewardCallback(logger_idx=1),  # first match, either player is fine
                 CheckpointCallback(save_freq,
                                    save_path=models_folder + "MLP_1_2x512_2_3x256",
                                    name_prefix="model")]
    model.learn(total_timesteps=1_000_000_000,
                callback=callbacks,
                tb_log_name="PPO_MLP_1_2x512_2_3x256",
                reset_num_timesteps=False
                )
    model.save(models_folder + "MLP_final")
    env.close()
