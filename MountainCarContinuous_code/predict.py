import gymnasium as gym
from stable_baselines3 import SAC, PPO, A2C, TD3
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
import cv2

def main():
    # 1. Define the environment using gymnasium
    env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array", goal_velocity=0.1)

    # 2. Vectorize the environment
    # (若有包含多个env的列表传入DummyVecEnv，可用一个线程执行多个env，提高训练效率)
    env = DummyVecEnv([lambda : env])

    ## 3. Load the trained model
    RL_NAME = 'SAC'
    CHECKPOINT = '125000'
    model = SAC.load("./model/MountainCarContinuous/"+RL_NAME+"_MountainCarContinuous_"+CHECKPOINT+".pkl")

    # 4. Simulate the interaction process between the agent and the environment
    for i in range(10):
        # 1) Initialize some variables
        observation = env.reset() # reset the environment to get the initial state
        score = 0
        done = False
        while not done:
            # 2) Agent policy that uses the observation and info
            action, _ = model.predict(observation=observation)
            # 3) Step (transition) through the environment with the action
            #   receiving the next observation, reward and if the episode has terminated
            #   done: whether we should stop the environment
            observation, reward, done, info = env.step(actions=action)
            # 4) compute the score
            score += reward
            # 5) Show rendering results using rgb_array mode and OpenCV
            frame = env.render()
            cv2.imshow('env', frame)
            cv2.waitKey(1)
            # 6) output the score
            if done:
                print(f"Episode finished with score: {score}")
                score = 0
                observation = env.reset()
                        
    ## 5. close the environment
    env.close()

if __name__ == '__main__':
    main()