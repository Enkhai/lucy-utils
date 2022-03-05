import rlgym
from rlgym.utils.reward_functions import common_rewards
from rlgym.utils.terminal_conditions.common_conditions import GoalScoredCondition, TimeoutCondition
from rlgym_tools.sb3_utils.sb3_log_reward import SB3CombinedLogReward, SB3CombinedLogRewardCallback
from stable_baselines3 import PPO

if __name__ == '__main__':
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

    #           obs
    #            |
    #          <128>
    #            |
    #          <128>
    #  policy         value
    # branch          branch
    #    /               \
    #  <64>              <64>
    #   |                 |
    #  <32>              <32>
    #   |                 |
    #  <16>             value
    #   |
    # action
    # The `net_arch` below describes the MLP network architecture presented above
    # stable_baselines3.common.policies.ActorCriticPolicy arguments
    policy_kwargs = dict(net_arch=[128, 128,  # shared layers
                                   dict(vf=[64, 32],  # value branch
                                        pi=[64, 32, 16])  # policy branch
                                   ])
    model = PPO(policy="MlpPolicy",
                env=env,
                policy_kwargs=policy_kwargs,
                tensorboard_log="./bin",
                verbose=1,
                device="cpu")  # CPU is faster for small networks such as this
    model.set_random_seed(0)

    reward_log_callback = SB3CombinedLogRewardCallback(reward_names=reward_names)
    model.learn(total_timesteps=100_000_000, callback=reward_log_callback)

    env.close()
