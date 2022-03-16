import rlgym
env = rlgym.make()

while True:
    obs = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()
        obs, reward, done, gameinfo = env.step(action)

# *This is never called in this case*
env.close()
