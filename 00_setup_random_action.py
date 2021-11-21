import rlgym

# Open up Bakkesmod first!
# Set up the environment with default arguments and launch the game
env = rlgym.make()

# Iteratively repeat
while True:
    # We retrieve the first observation from the initial game state
    obs = env.reset()
    done = False

    # While the game is not over
    while not done:
        # Sample a random action. If an agent was used, the action would be retrieved from it
        action = env.action_space.sample()
        # Take a step
        obs, reward, done, gameinfo = env.step(action)

# *This is never called in this case*
# Completed. Disconnect communication with Bakkesmod and close the game.
env.close()
