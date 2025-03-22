import gymnasium as gym
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import os

## 1. Define the environment using gymnasium
# 2. 使用 Monitor 包装环境
env = Monitor(gym.make('HumanoidStandup-v4', render_mode="human", width=1280, height=1024))

## 2. Vectorize the environment
## (若有包含多个env的列表传入DummyVecEnv，可用一个线程执行多个env，提高训练效率)
env = DummyVecEnv([lambda : env])

## 3. Define the model
model = PPO(
    "MlpPolicy", # the policy model to use
    env, # the environment to learn from
    verbose=1, # print info messages (such as device or wrappers used)
    tensorboard_log="./HumanoidStandup-v4/" # the log location for tensorboard
    
    # 👇PPO默认参数
    # "MlpPolicy", 
    # env,
    # learning_rate = 3e-4,
    # n_steps = 2048,
    # batch_size = 64,
    # n_epochs = 10,
    # gamma = 0.99,
    # gae_lambda = 0.95,
    # clip_range = 0.2,
    # clip_range_vf = None,
    # normalize_advantage = True,
    # ent_coef = 0.0,
    # vf_coef = 0.5,
    # max_grad_norm = 0.5,
    # use_sde = False,
    # sde_sample_freq = -1,
    # rollout_buffer_class = None,
    # rollout_buffer_kwargs = None,
    # target_kl = None,
    # stats_window_size = 100,
    # tensorboard_log = None,
    # policy_kwargs = None,
    # verbose = 0,
    # seed = None,
    # device = "auto",
    # _init_setup_model = True
    
    # 👇PPO推荐参数
#     "MlpPolicy", 
#     normalize=True,
#     n_envs= 1,
#   n_timesteps=1e7,
#   batch_size=32,
#   n_steps=512,
#   gamma=0.99,
#   learning_rate=2.55673e-05,
#   ent_coef=3.62109e-06,
#   clip_range=0.3,
#   n_epochs=20,
#   gae_lambda=0.9,
#   max_grad_norm=0.7,
#   vf_coef=0.430793,
#   policy_kwargs="dict(
#                     log_std_init=-2,
#                     ortho_init=False,
#                     activation_fn=nn.ReLU,
#                     net_arch=dict(pi=[256, 256], vf=[256, 256])
#                   )"
)

## 4. Train the model
# model.learn(total_timesteps=1e4, log_interval=4)
# model.learn(total_timesteps=2e6)
model.learn(total_timesteps=25000)

## 5. Evaluate the policy
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=True) # return the mean and variance of the model's scores after n tests
print("mean_reward: ", mean_reward, "; std_reward: ", std_reward)

## 6. Save the model
path = './model'
if not os.path.exists(path):
    os.makedirs("./model", exist_ok=True)
model.save("./model/HumanoidStandup.pkl")

## 7. Close the environment
env.close()
