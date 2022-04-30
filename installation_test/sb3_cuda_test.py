import rlgym
from lucy_utils.algorithms import DeviceAlternatingPPO
from rlgym.utils.terminal_conditions.common_conditions import GoalScoredCondition, TimeoutCondition

env = rlgym.make(terminal_conditions=[TimeoutCondition(7000), GoalScoredCondition()])
model = DeviceAlternatingPPO(policy="MlpPolicy", env=env, n_steps=32_000, verbose=1, device='cuda')

model.learn(total_timesteps=int(1e6))
env.close()
