import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle # serializing and deserializing python objects

def run(episodes, is_training=True, render=False): # default mode: training
    # 1. Define the environment using gymnasium
    env = gym.make('MountainCar-v0', render_mode='human' if render else None) # show the render when predicting

    # 2. State space (Divide position and velocity into 20 segments)
    pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 20) # between -1.2 and 0.6
    vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 20) # between -0.07 and 0.07

    # 3. Initialise the q-table
    if(is_training):
        # init a 20x20x3 array, representing three possible actions
        q = np.zeros((len(pos_space), len(vel_space), env.action_space.n))
    else:
        # load from the model file
        f = open('mountain_car_discrete.pkl', 'rb')
        q = pickle.load(f)
        f.close()

    # 4. Initialise other hyperparameters
    learning_rate_a = 0.9 # alpha or learning rate
    discount_factor_g = 0.9 # gamma or discount factor

    epsilon = 1 # randomly choose an action with probability epsilon
    epsilon_decay_rate = 2 / episodes # epsilon decay rate
    # epsilon is very large at the start of training.
    # as training progresses, the approximation of the Q-table will get better, 
    #   gradually reducing epsilon
    
    rng = np.random.default_rng() # random number generator
    rewards_per_episode = np.zeros(episodes) # record reward per episode

    # 5. Traverse each episode
    for i in range(episodes):
        # (step1) starting position, starting velocity always 0
        state = env.reset()[0]
        #   example result of env.reset(): (array([-0.4914937,  0.       ], dtype=float32), {})
        state_p = np.digitize(state[0], pos_space) # position
        state_v = np.digitize(state[1], vel_space) # velocity
        # true when reached goal
        terminated = False
        # initialise the rewards list
        rewards = 0
        
        while (not terminated and rewards > -1000):
            # (step2) choose an action using the epsilon-greedy strategy
            if is_training and rng.random() < epsilon:
                # choose random action (0=drive left, 1=stay neutral, 2=drive right)
                action = env.action_space.sample()
            else:
                # choose the best action
                action = np.argmax(q[state_p, state_v, :])
            # (step3) get new state (position and velocity), reward and terminated
            new_state, reward, terminated, _, _ = env.step(action)
            new_state_p = np.digitize(new_state[0], pos_space) # position
            new_state_v = np.digitize(new_state[1], vel_space) # velocity
            # (step4) update the Q-table when training
            if is_training:
                q[state_p, state_v, action] = q[state_p, state_v, action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state_p, new_state_v, :]) - q[state_p, state_v, action]
                )
            # update the state
            state = new_state
            state_p = new_state_p
            state_v = new_state_v
            # update overall reward
            rewards += reward
        # update epsilon
        epsilon = max(epsilon - epsilon_decay_rate, 0)
        # store rewards per episode
        rewards_per_episode[i] = rewards
        # output the reward each 100 epoches
        if i % 100 == 0 and is_training:
            print(f"Current training epoch: {i}, having reward of {rewards_per_episode[i]}.")
        if not is_training:
            print(f"Current testing epoch: {i + 1}, having reward of {rewards_per_episode[i]}.")
    
    # 6. Close the environment
    env.close()
    
    # 7. Save Q-table to file when training
    if is_training:
        f = open('mountain_car_discrete.pkl','wb')
        pickle.dump(q, f) # write the pickled representation of Q-table 'q' to the open file object 'f'
        f.close()
        
    # 8. Compute and plot the average reward for the past t rounds during training and testing (max 100 rounds)
    mean_rewards = np.zeros(episodes)
    for t in range(episodes):
        mean_rewards[t] = np.mean(rewards_per_episode[max(0, t - 100): (t + 1)])
    if is_training:
        label = 'Training mean rewards'
    else:
        label = 'Testing mean rewards'
    plt.plot(mean_rewards, label=label)
    plt.title("Average reward for the past t rounds (max 100 rounds)")
    plt.legend(loc='lower right')
    path = './result'
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    plt.savefig(f'result/mountain_car.png')
    print("Successfully plot and save the average reward!")

if __name__ == '__main__':
    # 1. Train
    print("Start training...")
    run(episodes=5000, is_training=True, render=False)
    print("End training!!!")
    # 2. Test
    print("Start testing...")
    run(episodes=10, is_training=False, render=True)
    print("End testing!!!")
