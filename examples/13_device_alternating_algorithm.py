import rlgym
from torch import cuda

# This example is dedicated to an algorithm that alternates between experience collection (rollouts) on the CPU
# and model training on the GPU
from utils.algorithms import DeviceAlternatingPPO

if __name__ == '__main__':
    assert cuda.is_available(), "Your PyTorch installation does not support CUDA operations. " \
                                "You cannot run this example."
    env = rlgym.make()
    model = DeviceAlternatingPPO(policy="MlpPolicy", env=env, verbose=1)
    model.learn(total_timesteps=int(1e6))
    env.close()
