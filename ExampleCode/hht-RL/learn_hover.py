'''Train hht-RL models with GUI to debug before real training. 
Using gym.make to build the env, in constract to use stablebaselines3'''

import sys

sys.path.append('/home/hht/simul0703/QY-hummingbird/')

from hitsz_qy_hummingbird.envs.rl_att_body import RLattbody
from hitsz_qy_hummingbird.envs.rl_attitude import RLatt
from hitsz_qy_hummingbird.configuration import configuration
from hitsz_qy_hummingbird.base_FWMAV.wings.wing_params import ParamsForBaseWing
from hitsz_qy_hummingbird.configuration.configuration import GLOBAL_CONFIGURATION

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO


class learn_hover:

    def __init__(self):
        None

    def run(self):

        #### Check the environment's spaces ########################
        env = gym.make("attbody-qyhb-v0")
        print("[INFO] Action space:", env.action_space)
        print("[INFO] Observation space:", env.observation_space)

        #### Train the model #######################################
        model = PPO("MlpPolicy",
                    env,
                    verbose=1,
                    device='cpu'
                    )
        model.learn(total_timesteps=10000)  #

        env = RLattbody(gui=True)

        obs, info = env.reset()

        for i in range(6 * env.CTRL_FREQ):
            action, _states = model.predict(obs,
                                            deterministic=True
                                            )
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs = env.reset()
        env.close()


if __name__ == "__main__":
    learn = learn_hover()
    learn.run()
