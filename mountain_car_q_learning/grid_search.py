import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import time
import pandas as pd
from scipy.stats import ttest_ind
import seaborn as sns
import itertools

# Constants for visualization
ACTION_SPACE = {
    0: 'Accelerate to the left',
    1: 'Don\'t accelerate',
    2: 'Accelerate to the right'
}

def check_directory():
    """Guarantee directories exist."""
    
    os.makedirs('./model', exist_ok=True)
    os.makedirs('./result', exist_ok=True)

def save_model(q_table, v_table, uuid):
    """Save Q-table and V-table to files."""
    
    with open(f"model/mountain_car_discrete_q_{uuid}.pkl", 'wb') as f:
        pickle.dump(q_table, f)
    with open(f"model/mountain_car_discrete_v_{uuid}.pkl", 'wb') as f:
        pickle.dump(v_table, f)

def load_model(uuid):
    """Load Q-table and V-table from files."""
    
    with open(f"model/mountain_car_discrete_q_{uuid}.pkl", 'rb') as f:
        q_table = pickle.load(f)
    with open(f"model/mountain_car_discrete_v_{uuid}.pkl", 'rb') as f2:
        v_table = pickle.load(f2)
    return q_table, v_table

def secondary_value_function(phi_table, state_p, state_v, new_p, new_v, 
                             action, reward, gamma):
    """Secondary value function for Reward Shaping."""
    
    # Learning rate for the secondary value function
    BETA = 0.5 / 14
    # Calculate the current value and the new value of the secondary value function
    phi_value = phi_table[state_p, state_v, action]
    new_phi_value = phi_table[new_p, new_v, action]
    # Timing Differential Error
    tde = -reward + gamma * new_phi_value - phi_value
    # Update the secondary value function
    phi_table[state_p, state_v, action] += BETA * tde

def calculate_shaped_reward(phi_table, state_p, state_v, new_p, new_v, 
                            action, reward, gamma):
    """Calculate the shaped reward."""
    
    # Calculate the secondary value function
    secondary_value_function(phi_table, state_p, state_v, new_p, new_v, 
                             action, reward, gamma)
    # Calculate the shaped reward
    shaped_reward = gamma * phi_table[new_p, new_v, action] \
                    - phi_table[state_p, state_v, action]
    return shaped_reward

def run(episodes, hp, is_training=True, render=False):
    """Main training/testing loop with Q-learning implementation."""
    
    # Parameters setup
    durations = []
    UUID = hp['UUID']
    SPC_POS_IDX = hp['spe_pos_idx']
    alpha = hp['alpha']
    gamma = hp['gamma']
    epsilon = hp['epsilon']
    epsilon_decay = hp['epsilon_decay_hp'] / episodes
    reward_mode = hp['reward_mode']
    rng = np.random.default_rng()
    rewards_log = np.zeros(episodes)
    cnt_success = 0
    
    # Environment setup
    env = gym.make('MountainCar-v0', 
                   render_mode='human' if render else None,
                   goal_velocity=hp['goal_velocity'])
    
    # State space discretization
    pos_bins = np.linspace(env.observation_space.low[0], env.observation_space.high[0], hp['pos_space_seg'])
    vel_bins = np.linspace(env.observation_space.low[1], env.observation_space.high[1], hp['vel_space_seg'])
    
    # Initialize Q/V/Phi tables
    if is_training:
        q_table = np.zeros((len(pos_bins), len(vel_bins), env.action_space.n))
        v_table = np.zeros((episodes, hp['pos_space_seg']))
        phi_table = np.zeros_like(q_table)
    else:
        q_table, v_table = load_model(UUID)
    
    # Traverse each episode
    for ep in range(episodes):
        # Variables initialization and environment reset
        start_time = time.perf_counter()
        state = env.reset()[0]
        state_p = np.digitize(state[0], pos_bins)
        state_v = np.digitize(state[1], vel_bins)
        terminated = False
        total_reward = 0
        sum_v = np.zeros(hp['pos_space_seg']) # V-table related
        cnt_v = np.zeros_like(sum_v) # V-table related
        success = False
        
        while not terminated and total_reward > -1000:
            # Epsilon-greedy action selection
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state_p, state_v, :])

            # Environment step
            new_state, reward, terminated, _, _ = env.step(action)
            new_p = np.digitize(new_state[0], pos_bins)
            new_v = np.digitize(new_state[1], vel_bins)

            if is_training:
                # Reward setup according to the reward mode
                extra_reward = 0 if reward_mode == 1 else calculate_shaped_reward(
                    phi_table, state_p, state_v, new_p, new_v, action, reward, gamma
                )
                # Q-table update
                q_table[state_p, state_v, action] += alpha * (
                    reward + extra_reward 
                    + gamma * np.max(q_table[new_p, new_v, :]) 
                    - q_table[state_p, state_v, action]
                )
                # V-table update
                cnt_v[new_p] += 1
                sum_v[new_p] += vel_bins[new_v]

            # Success condition check
            if new_state[0] > env.unwrapped.goal_position: # goal_position: destination's axis, a constant (0.5)
                success = True
                cnt_success += 1
            
            # State and rewards update
            state_p, state_v = new_p, new_v
            total_reward += reward

        # Calculate the average velocity for each position state
        for idx in range(len(sum_v)):
            if cnt_v[idx] != 0:
                v_table[ep, idx] = sum_v[idx] / cnt_v[idx]

        # Post-episode updates
        epsilon = max(epsilon - epsilon_decay, 0)
        rewards_log[ep] = total_reward
        durations.append(time.perf_counter() - start_time)

        # Progress reporting
        phase = 'training' if is_training else 'testing'
        ratio = 100 if is_training else 1
        if ep % ratio == 0:
            print(f"Current {phase} epoch: {ep}, Reward: {total_reward:.1f}, Success: {'Yes' if success else 'No'}")

    # Environment close
    env.close()

    # Check directory existence
    check_directory()

    # Save models and generate visualizations
    if is_training:
        save_model(q_table, v_table, UUID)
        plot_duration_boxplot(durations, UUID, batch_size=100)
        # V-table visualization
        plot_linechart(v_table[:, SPC_POS_IDX], episodes, SPC_POS_IDX, UUID)
    else:
        for action in range(env.action_space.n):
            plot_heatmap(q_table[:, :, action], UUID, action, 'q')

    # Calculate moving average rewards
    window_size = 100 if is_training else 1
    mean_rewards = np.convolve(rewards_log, np.ones(window_size) / window_size, mode='valid')
    plot_mean_rewards(mean_rewards, UUID, is_training)

    # Calculate overall duration for current experiment
    sum_duration = sum(durations)

    # Calculate success rate
    success_rate = cnt_success / episodes if episodes != 0 else 0

    # Calculate overall mean rewards during the whole training/testing process
    avg_rewards = np.mean(mean_rewards[:-100]) if is_training else np.mean(mean_rewards)

    return mean_rewards, avg_rewards, sum_duration, success_rate

def plot_duration_boxplot(durations, uuid, batch_size=100):
    """Plot and save duration distribution across training batches."""
    
    # Split durations into batches
    batches = [durations[i * batch_size: (i + 1) * batch_size] 
               for i in range(len(durations) // batch_size)]
    
    plt.figure(figsize=(16, 6))
    plt.boxplot(batches)
    plt.title(f"Training Duration Distribution [Version: {uuid}]")
    plt.ylabel("Time (seconds)")
    plt.xlabel("Training Batches (100 episodes/batch)")
    plt.tight_layout()
    plt.savefig(f'result/mountain_car_duration_{uuid}.png', dpi=300)
    plt.close()
    print("Saved training duration distribution chart!")

def plot_mean_rewards(mean_rewards, uuid, is_training=True):
    """Plot and save average rewards visualization."""
    
    plt.figure(figsize=(7, 5))
    label_type = 'Training' if is_training else 'Testing'
    plt.plot(mean_rewards, label=f'{label_type} mean rewards')
    plt.title(f"Moving Average Reward [Version: {uuid}]")
    plt.legend(loc='lower right' if is_training else 'upper right')
    plt.savefig(f'result/mountain_car_{label_type.lower()}_reward_{uuid}.png', dpi=300)
    plt.close()
    print(f"Saved {label_type.lower()} reward chart!")

def plot_linechart(data, episodes, pos_idx, uuid):
    """Plot and save average velocity visualization."""
    
    plt.figure(figsize=(22, 8))
    plt.plot(range(episodes), data)
    plt.xlabel("Episode")
    plt.ylabel("Velocity")
    plt.title(f"Average Velocity Line Chart For Position State {pos_idx} [Version: {uuid}]")
    plt.savefig(f'result/mountain_car_v_{uuid}.png', dpi=300)
    plt.close()
    print(f"Saved average velocity line chart for position state {pos_idx}!")

def plot_heatmap(data, uuid, action_idx, data_type='q'):
    """Plot and save heatmap visualization."""
    
    plt.figure(figsize=(24, 8) if data_type == 'q' else (16, 8))
    ax = sns.heatmap(data=data, annot=True, fmt=".3f", linewidth=0.5, cbar=True, cmap="crest")
    ax.set(xlabel="Position State", ylabel="Velocity State")
    ax.xaxis.tick_top()
    action_desc = ACTION_SPACE[action_idx]
    plt.title(f"Heatmap For {data_type.upper()}-table (Action {action_idx}-{action_desc}) [Version: {uuid}]")
    plt.savefig(f'result/mountain_car_{data_type}{action_idx}_{uuid}.png', dpi=300)
    plt.close()
    print(f"Saved {data_type}-table heatmap for action {action_idx}!")

def run_experiment(hp_config):
    """Complete training/testing cycle for one configuration."""
    
    print(f"\n=== Starting Experiment {hp_config['UUID']} ===")
    train_rewards, avg_train_rewards, train_duration, train_success_rate = run(
        hp_config['train_epoch'], hp_config, is_training=True, render=False
    )
    print()
    test_rewards, avg_test_rewards, test_duration, test_success_rate = run(
        hp_config['test_epoch'], hp_config, is_training=False, render=True
    )
    return train_rewards, avg_train_rewards, train_duration, train_success_rate, \
        test_rewards, avg_test_rewards, test_duration, test_success_rate

def grid_search(grid):
    """Perform grid search for hyperparameter optimization."""
    
    # Check whether training epochs are variable
    param_variable = False
    
    # Identification of variable hyperparameters
    keys_list = list(grid.keys()) # To get key's name
    param_list = [] # Hyperparameters having multiple values
    for i, v in enumerate(grid.values()):
        if len(v) > 1:
            param_list.append(keys_list[i])
            if keys_list[i] == 'train_epoch': # Training epochs are variable
                param_variable = True
    
    # Hyperparameter configuration combination
    configs = []
    for combination in itertools.product(*grid.values()): # Generate cartesian product
        # Current hyperparameter configuration
        param = dict(zip(grid.keys(), combination))
        
        # UUID generation
        UUID = ''
        for i, key in enumerate(param_list):
            UUID += key + '_' + str(param[key])
            if i != len(param_list) - 1:
                UUID += '&'
        if UUID == '':
            UUID = 'default'
        param['UUID'] = UUID

        # Append constructed hyperparameter configuration
        configs.append(param)

    return configs, param_variable

if __name__ == '__main__':
    # Hyperparameter grid definition
    grid = {
        'train_epoch': [5900, 6000],
        'test_epoch': [10],
        'goal_velocity': [0],
        'pos_space_seg': [20],
        'vel_space_seg': [20],
        'alpha': [0.3],
        'gamma': [0.37],
        'epsilon': [0.3],
        'epsilon_decay_hp': [2.5],
        'spe_pos_idx': [8],
        'reward_mode': [2]
    }

    # Grid Search
    configs, param_variable = grid_search(grid)
    
    # Run experiments
    results = {}
    for cfg in configs:
        train_rewards, avg_train_rewards, train_duration, train_success_rate, \
            test_rewards, avg_test_rewards, test_duration, test_success_rate = run_experiment(cfg)
        results[cfg['UUID']] = {
            'Hyperparameter': cfg,
            'Training rewards': train_rewards,
            'Average training rewards': avg_train_rewards,
            'Duration of training': train_duration,
            'Training success rate': train_success_rate,
            'Testing rewards': test_rewards,
            'Average testing rewards': avg_test_rewards,
            'Duration of testing': test_duration,
            'Testing success rate': test_success_rate
        }
    
    # Statistical analysis (T-test)
    df = pd.DataFrame(results)
    df.to_excel("result/mountain_car_experiments.xlsx")
    if len(results) > 1:
        # Perform pairwise T-tests
        for cfg1, cfg2 in itertools.combinations(results.keys(), 2):
            train_rewards_a = df[cfg1]['Training rewards']
            train_rewards_b = df[cfg2]['Training rewards']
            ttest_result = ttest_ind(train_rewards_a, train_rewards_b)
            print(f"\nT-test Results:\nThe 1st configuration: {cfg1}.\nThe 2nd configuration: {cfg2}.")
            print(f"T-statistic: {ttest_result.statistic:.3f}.\nP-value: {ttest_result.pvalue:.4f}.")
    else:
        print("\nT-test needs two hyperparameter configurations.")
    
    # Best hyperparameter configuration
    # NOTE: This result is a reference for confirming the optimal configuration, which needs manual inspection.
    print(f"\nAre there any variable hyperparameters? {param_variable}")
    
    best_idx = -1
    best_cfg = {'Average testing rewards': -np.inf, 'Duration of training': np.inf, 'Testing success rate': -np.inf}
    for idx, cfg in enumerate(results.values()):
        con1 = cfg['Average testing rewards'] > best_cfg['Average testing rewards']
        con2 = cfg['Duration of training'] < best_cfg['Duration of training']
        con3 = cfg['Testing success rate'] >= best_cfg['Testing success rate']
        con = con1 and con2 and con3 if not param_variable else con1 and con3
        # if con1 and con2 and con3:
        if con:
            best_idx = idx
            best_cfg = cfg
    print(f"\nBest hyperparameter configuration: ")
    for param in best_cfg.keys():
        print(f"{param}: {best_cfg[param]}")
