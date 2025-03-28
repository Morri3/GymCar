import gymnasium as gym
from stable_baselines3 import SAC, PPO, TD3
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.noise import NormalActionNoise
import os
import torch.nn as nn
import torch
import numpy as np
print(torch.cuda.is_available())

## 1. Define the environment using gymnasium
env = Monitor(gym.make('Humanoid-v4', render_mode="human", width=1280, height=1024))

## 2. Vectorize the environment
## (若有包含多个env的列表传入DummyVecEnv，可用一个线程执行多个env，提高训练效率)
env = DummyVecEnv([lambda : env])

## 4. Define the model
RL_NAME = 'SAC'
model = SAC(
    "MlpPolicy", # the policy model to use
    env, # the environment to learn from
    verbose = 1, # print info messages (such as device or wrappers used)
    tensorboard_log = "./Humanoid-v4/", # the log location for tensorboard
    # learning_rate = 3e-5,
    device="cuda" # using gpu
)
# # 1) The noise objects for TD3
# n_actions = env.action_space.shape[-1]
# action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
# # 2) Define the model
# model = TD3(
#     "MlpPolicy", # the policy model to use
#     env, # the environment to learn from
#     verbose = 1, # print info messages (such as device or wrappers used)
#     tensorboard_log = "./Humanoid-v4/", # the log location for tensorboard
#     # learning_rate = 3e-5,
#     action_noise=action_noise, # TD3
#     device="cuda" # using gpu
# )
# model = PPO(
#     "MlpPolicy", # the policy model to use
#     env, # the environment to learn from
#     verbose = 1, # print info messages (such as device or wrappers used)
#     tensorboard_log = "./Humanoid-v4/", # the log location for tensorboard
#     # learning_rate = 3e-5,
#     learning_rate = 3.56987e-05,
#     n_steps = 512,
#     batch_size = 256,
#     n_epochs = 5,
#     gamma = 0.95,
#     gae_lambda = 0.9,
#     clip_range = 0.3,
#     normalize_advantage = True,
#     ent_coef = 0.00238306,
#     vf_coef = 0.431892,
#     max_grad_norm = 2,
#     policy_kwargs = dict(
#                         log_std_init=-2,
#                         ortho_init=False,
#                         activation_fn=nn.ReLU,
#                         net_arch=dict(pi=[256, 256], vf=[256, 256])
#                     )
#     # device="cuda" # using gpu
# )

# ## 4. Train the model
# # 1) define the callback
# eval_callback = EvalCallback(env, best_model_save_path='./best_models/', log_path='./logs/', verbose=1, eval_freq=1000)
# # checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./logs/', name_prefix='PPO')
# # 2) train the model
# # model.learn(total_timesteps=25000, callback=eval_callback)
# model.learn(total_timesteps=2e6, callback=eval_callback)
# # model.learn(total_timesteps=1e4, log_interval=4)

## 4. Save the model
path = './model'
if not os.path.exists(path):
    os.makedirs("./model", exist_ok=True)
    
## 5. Train the model
TOTAL_TIMESTEPS = 25000
for i in range(15):
    # 1) define the callback
    eval_callback = EvalCallback(env, best_model_save_path='./best_models/'+RL_NAME+'/', log_path='./logs/', verbose=1, eval_freq=1000)
    # checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./logs/', name_prefix='PPO')
    # 2) train the model
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)
    # model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback, reset_num_timesteps=False)
    # 3) save the model
    model.save(f"./model/{RL_NAME}_Humanoid_{TOTAL_TIMESTEPS*(i+1)}.pkl")

## 6. Evaluate the policy
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=True) # return the mean and variance of the model's scores after n tests
print("mean_reward: ", mean_reward, "; std_reward: ", std_reward)

## 7. Close the environment
env.close()
