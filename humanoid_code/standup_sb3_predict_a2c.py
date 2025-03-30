import gymnasium as gym
from stable_baselines3 import SAC, PPO, A2C
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
import cv2
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

def main():
    ## 1. Define the environment
    env = make_vec_env('Humanoid-v4', n_envs=2, vec_env_cls=SubprocVecEnv)

    ## 2. Load the trained model
    RL_NAME = 'A2C'
    CHECKPOINT = '1500000'
    model = A2C.load("./model/"+RL_NAME+"_Humanoid_"+CHECKPOINT+".pkl")

    ## 3. Initialize some variables
    observation = env.reset() # reset the environment to get the initial state
    score = 0
    done = False

    ## 4. Simulate the interaction process between the agent and the environment
    obs = env.reset()
    while True:
        # 1) Agent policy that uses the observation and info
        action, _ = model.predict(observation=observation)
        # 2) Step (transition) through the environment with the action
        #   receiving the next observation, reward and if the episode has terminated
        #   done: whether we should stop the environment
        observation, reward, done, info = env.step(actions=action)
        # 3) compute the score
        score += reward
        env.render("human")
                
    ## 5. close the environment
    env.close()

if __name__ == '__main__':
    main()