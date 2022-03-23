import rlgym
from rlgym_tools.sb3_utils.sb3_single_instance_env import SB3SingleInstanceEnv
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from rlgym.utils.reward_functions import DefaultReward


def load_and_evaluate(model_path,
                      team_size,
                      terminal_conditions,
                      obs_builder,
                      state_setter,
                      action_parser,
                      reward_fn=DefaultReward(),
                      self_play=True,
                      game_speed=1,
                      iterations=10,
                      device='cpu',
                      tick_skip=8):
    """
    Creates a single-instance environment for evaluation
    """
    iterations *= team_size * (self_play + 1)

    env = rlgym.make(game_speed=game_speed,
                     tick_skip=tick_skip,
                     self_play=self_play,
                     team_size=team_size,
                     terminal_conditions=terminal_conditions,
                     reward_fn=reward_fn,
                     obs_builder=obs_builder,
                     action_parser=action_parser,
                     state_setter=state_setter)
    env = SB3SingleInstanceEnv(env)

    model = PPO.load(model_path, env, device)

    mean_rew, rew_std = evaluate_policy(model, env, iterations)
    print("Mean reward: {}, Reward standard deviation: {}".format(mean_rew, rew_std))

    env.close()
