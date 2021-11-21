import rlgym
# There are various common conditions available for RLGym. You can check those in `common_conditions`
from rlgym.utils.terminal_conditions.common_conditions import GoalScoredCondition, TimeoutCondition
# Additionally, other common classes can be found in `utils` with the `common_` prefix
# We import the Proximal Policy Optimization model
from stable_baselines3 import PPO

# We make the environment with our own terminal conditions this time
env = rlgym.make(terminal_conditions=[TimeoutCondition(7000), GoalScoredCondition()])

# To watch GPU memory consumption you can open a new terminal while the game is running and type `nvidia-smi -l 1`
# If you have Cygwin installed you can instead type `watch -n 1 nvidia-smi` for a continuous and improved GPU
# state logging
# If you have multiple GPUs in your system you can also have the game running on the less
# capable GPU (Nvidia Control Panel) and the model on the more capable one `(device='cuda:<device number>')`
# `verbose=1` logs back training info
model = PPO(policy="MlpPolicy", env=env, verbose=1, device='cuda')

model.learn(total_timesteps=int(1e6))

# After learning is complete disconnect communication to Bakkesmod and close the game
env.close()
