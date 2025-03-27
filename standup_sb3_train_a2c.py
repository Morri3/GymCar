import gymnasium as gym
from stable_baselines3 import SAC, PPO, A2C
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
import os
import torch.nn as nn
import torch

def main():
    print(torch.cuda.is_available())

    ## 1. Define the environment using gymnasium
    # env = Monitor(gym.make('Humanoid-v4', render_mode="human", width=1280, height=1024))
    env = make_vec_env('Humanoid-v4', n_envs=2, vec_env_cls=SubprocVecEnv)

    ## 2. Define the model
    RL_NAME = 'A2C'
    model = A2C(
        "MlpPolicy", # the policy model to use
        env, # the environment to learn from
        verbose = 1, # print info messages (such as device or wrappers used)
        tensorboard_log = "./Humanoid-v4/", # the log location for tensorboard
        device = 'cpu'
    )

    ## 3. Save the model
    path = './model'
    if not os.path.exists(path):
        os.makedirs("./model", exist_ok=True)
        
    ## 4. Train the model
    TOTAL_TIMESTEPS = 25000
    for i in range(40):
        # 1) define the callback
        eval_callback = EvalCallback(env, best_model_save_path='./best_models/'+RL_NAME+'/', log_path='./logs/', verbose=1, eval_freq=1000)
        # checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./logs/', name_prefix='PPO')
        # 2) train the model
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)
        # model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback, reset_num_timesteps=False)
        # 3) save the model
        model.save(f"./model/{RL_NAME}_Humanoid_{TOTAL_TIMESTEPS*(i+1)}.pkl")

    ## 5. Evaluate the policy
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=True) # return the mean and variance of the model's scores after n tests
    print("mean_reward: ", mean_reward, "; std_reward: ", std_reward)

    ## 6. Close the environment
    env.close()

if __name__ == '__main__':
    main()
