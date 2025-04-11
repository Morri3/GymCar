import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle # serializing and deserializing python objects
import time
import pandas as pd
from scipy.stats import ttest_ind # t-test
import seaborn as sns
import math

def run(episodes, hp, is_training=True, render=False):
    # default mode: training
    # hp: hyperparameter configuration
    
    # 1. Initialise for counting
    durations = []
    UUID = hp['UUID']
    
    # 2. Define the environment using gymnasium
    env = gym.make('MountainCar-v0', render_mode='human' if render else None, goal_velocity=hp['goal_velocity']) # show the render only when predicting

    # 3. State space (Divide position and velocity into segments)
    pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], hp['pos_space_seg']) # between -1.2 and 0.6
    vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], hp['vel_space_seg']) # between -0.07 and 0.07

    # 4. Initialise the q-table and v-table
    if(is_training):
        # init a 20x20x3 array, representing Q-table for three possible actions
        q = np.zeros((len(pos_space), len(vel_space), env.action_space.n))
        # init a 20x20x3 array, representing v-table for three possible actions        
        v = np.zeros((len(pos_space), len(vel_space), env.action_space.n))
    else:
        # load from the model file (Q-table)
        f = open(f"model/mountain_car_discrete_q_{UUID}.pkl", 'rb')
        q = pickle.load(f)
        f.close()
        # load from the model file (v-table)
        f2 = open(f"model/mountain_car_discrete_v_{UUID}.pkl", 'rb')
        v = pickle.load(f2)
        f2.close()

    # 5. Initialise other hyperparameters
    learning_rate_a = hp['learning_rate_a'] # alpha or learning rate
    discount_factor_g = hp['discount_factor_g'] # gamma or discount factor

    epsilon = hp['epsilon'] # randomly choose an action with probability epsilon
    epsilon_decay_rate = hp['epsilon_decay_hp'] / episodes # epsilon decay rate
    # epsilon is very large at the start of training.
    # as training progresses, the approximation of the Q-table will get better, gradually reducing epsilon.
    
    rng = np.random.default_rng() # random number generator
    rewards_per_episode = np.zeros(episodes) # record reward per episode

    # 6. Traverse each episode
    for i in range(episodes):
        # get the start time
        start_time = time.perf_counter()
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
                # q-table
                q[state_p, state_v, action] = q[state_p, state_v, action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state_p, new_state_v, :]) - q[state_p, state_v, action]
                )
                # v-table, storing velocity for each (pos, vel) pairs (follow transition dynamics)
                force = 0.001
                gravity = 0.0025
                v[new_state_p, new_state_v, action] = v[state_p, state_v, action] + (action - 1) * force - math.cos(3 * state_p) * gravity
                # v[new_state_p, new_state_v, action] = new_state[1] # another method
                v[new_state_p, new_state_v, action] = np.clip(v[new_state_p, new_state_v, action], -0.07, 0.07) # clip to [-0.07, 0.07]
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
        
        # output the reward each 100 epochs
        if i % 100 == 0 and is_training:
            print(f"Current training epoch: {i}, having reward of {rewards_per_episode[i]}.")
        if not is_training:
            print(f"Current testing epoch: {i + 1}, having reward of {rewards_per_episode[i]}.")
        
        # get the end time
        end_time = time.perf_counter()
        # compute the duration and append into the list
        duration = end_time - start_time
        durations.append(duration)
        
    # 7. Close the environment
    env.close()
    
    # 8. Save Q-table to file when training
    if is_training:
        path_model = './model'
        if not os.path.exists(path_model):
            os.makedirs(path_model, exist_ok=True)
        # write the pickled representation of Q-table 'q' to the open file object 'f'
        f = open(f"model/mountain_car_discrete_q_{UUID}.pkl", 'wb')
        pickle.dump(q, f)
        f.close()
        # write the pickled representation of v-table 'v' to the open file object 'f2'
        f2 = open(f"model/mountain_car_discrete_v_{UUID}.pkl", 'wb')
        pickle.dump(v, f2)
        f2.close()
    
    # 9. Compute and plot the average reward for the past t rounds during training and testing (max 100 rounds)
    path_result = './result'
    if not os.path.exists(path_result):
        os.makedirs(path_result, exist_ok=True)
    
    mean_rewards = np.zeros(episodes)
    for t in range(episodes):
        mean_rewards[t] = np.mean(rewards_per_episode[max(0, t - 100): (t + 1)])
    if is_training:
        label = 'Training mean rewards'
    else:
        label = 'Testing mean rewards'
    plt.figure(figsize=(7, 5))
    plt.plot(mean_rewards, label=label)
    plt.title(f"Average Reward For The Past 100 Epochs [Version: {UUID}]")
    if is_training:
        plt.legend(loc='lower right')
        plt.savefig(f'result/mountain_car_train_reward_{UUID}.png', dpi=300)
    else:
        plt.legend(loc='upper right')
        plt.savefig(f'result/mountain_car_test_reward_{UUID}.png', dpi=300)
    print("Successfully plot and save the average reward!")
    
    action_space = {0: 'Accelerate to the left', 1: 'Don\'t accelerate', 2: 'Accelerate to the right'}
    if is_training:
        # 10. Plot the box chart for durations of each batch (100 epochs per batch)
        batch_size = 100
        duration_100_batch = [] # 2-d array
        for i in range(int(episodes / batch_size)):
            duration_100_batch.append(durations[(i * batch_size): ((i + 1) * batch_size)])
        
        plt.figure(figsize=(16, 6))
        plt.boxplot(duration_100_batch)
        plt.title(f"Duration For Each Batch (100 Epochs Per Batch) [Version: {UUID}]")
        plt.ylabel("Consuming Time (seconds)")
        plt.xlabel("Batch")
        plt.tight_layout()
        plt.savefig(f'result/mountain_car_duration_{UUID}.png', dpi=300)
        print("Successfully plot and save the duration for training!")
        
        # 11. Plot the 2-d heatmap for v-table (considering each action)
        for i in range(env.action_space.n):
            plt.figure(figsize=(16, 8))
            v_value = v[:, :, i]
            ax = sns.heatmap(data=v_value, annot=True, fmt=".3f", linewidth=.5, cbar=True, cmap="crest")
            ax.set(xlabel="position state", ylabel="velocity state")
            ax.xaxis.tick_top()
            plt.title(f"Heatmap For Velocity (Action {i}-{action_space[i]}) [Version: {UUID}]")
            plt.savefig(f'result/mountain_car_v{i}_{UUID}.png', dpi=300)
            print(f"Successfully plot and save the heatmap for velocity with action {i} [{action_space[i]}]!")
    else:
        # 12. Plot the 2-d heatmap for Q-table (considering each action)
        for i in range(env.action_space.n):
            plt.figure(figsize=(24, 8))
            q_value = q[:, :, i]
            ax = sns.heatmap(data=q_value, annot=True, fmt=".3f", linewidth=.5, cbar=True, cmap="crest")
            ax.set(xlabel="position state", ylabel="velocity state")
            ax.xaxis.tick_top()
            plt.title(f"Heatmap For Q-table (Action {i}-{action_space[i]}) [Version: {UUID}]")
            plt.savefig(f'result/mountain_car_q{i}_{UUID}.png', dpi=300)
            print(f"Successfully plot and save the heatmap for Q-table with action {i} [{action_space[i]}]!")
    
    # 13. show charts
    plt.show()
    return mean_rewards

def runOnce(hp):
    # 1. Train
    print("<=  Start training  =>")
    train_mean_rewards = run(episodes=hp['train_epoch'], hp=hp, is_training=True, render=False)
    print("<=  End training!  =>\n")
    
    # 2. Test
    print("<=  Start testing  =>")
    test_mean_rewards = run(episodes=hp['test_epoch'], hp=hp, is_training=False, render=True)
    print("<=  End testing!  =>")
    
    return train_mean_rewards, test_mean_rewards

if __name__ == '__main__':
    # 1. Construct two hyperparameter configurations
    HP = {
        'UUID': 'lr_0.9', 'train_epoch': 5000, 'test_epoch': 10, 'goal_velocity': 0, # UUID: to distinguish storage directories
        'pos_space_seg': 20, 'vel_space_seg': 20, 'learning_rate_a': 0.9, 'discount_factor_g': 0.9, 
        'epsilon': 1, 'epsilon_decay_hp': 2 # epsilon_decay_hp: epsilon decay hyperparameter
    }
    HP2 = {
        'UUID': 'lr_0.7', 'train_epoch': 5000, 'test_epoch': 10, 'goal_velocity': 0, # UUID: to distinguish storage directories
        'pos_space_seg': 20, 'vel_space_seg': 20, 'learning_rate_a': 0.7, 'discount_factor_g': 0.9, 
        'epsilon': 1, 'epsilon_decay_hp': 2 # epsilon_decay_hp: epsilon decay hyperparameter
    }
    
    # 2. Initialise the dictionary for conducting t-test
    results_table = {}
    
    # 3. Run experiments according to different hyperparameter configurations
    conf_list = [HP, HP2]
    for hp in list(enumerate(conf_list)):
        # hp[0]: index (start from 0)
        # hp[1]: content of the hyperparameter configuration
        print(f"<<<===  Start running experiment {hp[0] + 1}  ===>>>")
        print(f"Hyperparameter Configuration [{hp[0] + 1}]: \n{hp[1]}\n")
        train_mean_rewards, _ = runOnce(hp[1]) # run experiment once
        results_table[hp[0]] = train_mean_rewards # store results
        print(f"<<<===  End running experiment {hp[0] + 1}!  ===>>>\n\n")
    
    # 4. Conduct the t-test
    results = pd.DataFrame(results_table) # convert into a DataFrame and print
    results.to_excel("result/mountain_car_experiments.xlsx") # save as a xlsx
    print(f"Dataframe of result: \n{results}")
    print(f"Result of t-test: \n{ttest_ind(results[0], results[1])}") # t-test
