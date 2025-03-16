import gymnasium as gym

# 1. Initialise the environment
env = gym.make('HumanoidStandup-v5', impact_cost_weight=0.5e-6, \
    render_mode="human", width=1920, height=1280)
# 2. Reset the environment to generate the first observation and get extra info
observation, info = env.reset()
episode_over = False

while not episode_over:
    # 1) Agent policy that uses the observation and info
    #   ( this is where you would insert your policy )
    action = env.action_space.sample() # here is the random action
    
    # 2) Step (transition) through the environment with the action
    #   receiving the next observation, reward and if the episode has terminated
    #   terminated: bot has finished tasks or bot has been destroyed → stop the environment
    #   truncated: after a fixed number of timesteps → stop the environment
    observation, reward, terminated, truncated, info = env.step(action)

    # 3) Determine whether the episode is over
    episode_over = terminated or truncated
# 3. Close the environment
env.close()
