from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv
from rlgym_tools.sb3_utils.sb3_instantaneous_fps_callback import SB3InstantaneousFPSCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor

from obs import NectoObs
from reward import NectoRewardFunction
from utils.algorithms import DeviceAlternatingPPO
from utils.multi_instance_utils import get_matches, config
from utils.policies import ActorCriticAttnPolicy
from utils.rewards.sb3_log_reward import SB3NamedLogRewardCallback
from experiment.lucy_match_params import LucyTerminalConditions, LucyState, LucyAction

models_folder = "models_folder/"

# TODO: leave-one-out testing methodology - run for ~500 million steps and
#  replace each iteratively with a Lucy class object:
#  - model: ?
#  - observation: OK
#  - reward: ?


if __name__ == '__main__':
    num_instances = 8
    agents_per_match = 2 * 2  # self-play
    n_steps, batch_size, gamma, fps, save_freq = config(num_instances=num_instances,
                                                        avg_agents_per_match=agents_per_match,
                                                        target_steps=256_000,
                                                        target_batch_size=0.5,
                                                        callback_save_freq=10)

    matches = get_matches(reward_cls=NectoRewardFunction,
                          terminal_conditions=lambda: LucyTerminalConditions(fps),
                          obs_builder_cls=NectoObs,
                          action_parser_cls=LucyAction,
                          state_setter_cls=LucyState,
                          sizes=[agents_per_match // 2] * num_instances  # self-play, hence // 2
                          )

    env = SB3MultipleInstanceEnv(match_func_or_matches=matches)
    env = VecMonitor(env)

    policy_kwargs = dict(net_arch=[dict(
        # minus one for the key padding mask
        query_dims=env.observation_space.shape[-1] - 1,
        # minus eight for the previous action
        kv_dims=env.observation_space.shape[-1] - 1 - 8,
        n_preprocess_layers=2
    )] * 2)  # *2 because actor and critic will share the same architecture

    # model = DeviceAlternatingPPO.load("./models_folder/Perceiver/model_743680000_steps.zip", env)
    model = DeviceAlternatingPPO(policy=ActorCriticAttnPolicy,
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
                 CheckpointCallback(save_freq,
                                    save_path=models_folder + "NectoTest_Perceiver",
                                    name_prefix="model")]
    model.learn(total_timesteps=1_000_000_000,
                callback=callbacks,
                tb_log_name="NectoTest_PPO_Perceiver2_2x128",
                # reset_num_timesteps=False
                )
    model.save(models_folder + "NectoTest_Perceiver_final")

    env.close()
