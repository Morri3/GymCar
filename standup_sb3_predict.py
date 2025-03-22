import gymnasium as gym
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
import cv2

## 1. Define the environment using gymnasium
env = gym.make('HumanoidStandup-v4', render_mode="rgb_array", width=1280, height=1024)

## 2. Vectorize the environment
## (若有包含多个env的列表传入DummyVecEnv，可用一个线程执行多个env，提高训练效率)
env = DummyVecEnv([lambda : env])

## 3. Load the trained model
model = PPO.load("./model/HumanoidStandup.pkl")

## 4. Initialize some variables
observation = env.reset() # reset the environment to get the initial state
score = 0
done = False

## 5. Simulate the interaction process between the agent and the environment
while not done:
    # 1) Agent policy that uses the observation and info
    action, _ = model.predict(observation=observation)
    # 2) Step (transition) through the environment with the action
    #   receiving the next observation, reward and if the episode has terminated
    #   done: whether we should stop the environment
    observation, reward, done, info = env.step(actions=action)
    # 3) compute the score
    score += reward
    # 4) Show rendering results using rgb_array mode and OpenCV
    frame = env.render()
    cv2.imshow('env', frame)
    cv2.waitKey(1)

## 6. close the environment
env.close()
print("Final score: ", score)
