'''Continuing training on the old model allows for modifications to the reward function and certain model parameters, 
but does not permit alterations to the actions and state space.'''
import sys

sys.path.append('/home/hht/simul0703/QY-hummingbird/')

import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
from cycler import cycler

from hitsz_qy_hummingbird.envs.rl_attitude import RLatt
from hitsz_qy_hummingbird.envs.rl_flip import RLflip
from hitsz_qy_hummingbird.envs.rl_escape import RLescape

from hitsz_qy_hummingbird.configuration import configuration
from hitsz_qy_hummingbird.base_FWMAV.wings.wing_params import ParamsForBaseWing
from hitsz_qy_hummingbird.configuration.configuration import GLOBAL_CONFIGURATION

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy

DEFAULT_OUTPUT_FOLDER = 'att_results'

class LearnHoverS3():
    def __init__(self):
        self.output_folder = DEFAULT_OUTPUT_FOLDER
        self.infilename = os.path.join(self.output_folder, 'save-05.04.2024_12.10.54')
        self.outfilename = os.path.join(self.output_folder, 'save-' + datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
        if not os.path.exists(self.outfilename):
            os.makedirs(self.outfilename+'/')

    def load_model(self,env_s):

        self.model = PPO.load(self.infilename+'/best_model.zip',
                              env=env_s,
                              batch_size=512, 
                              gamma=0.995, 
                              device="cuda",)

    
    def train(self, myenv):
        
        train_env = make_vec_env(myenv,
                                 n_envs=20,
                                 seed=0,
                                 vec_env_cls=SubprocVecEnv)
        
        self.load_model(train_env)

        eval_env = make_vec_env(myenv,
                                n_envs=20,
                                seed=0,
                                vec_env_cls=SubprocVecEnv)

        # Check the environment's spaces
        print('[INFO] Action space:', train_env.action_space)
        print('[INFO] Observation space:', train_env.observation_space)

        target_reward = 1e10
        callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=target_reward,
                                                         verbose=1)
        eval_callback = EvalCallback(eval_env,
                                     callback_on_new_best=callback_on_best,
                                     verbose=1,
                                     best_model_save_path=self.outfilename+'/',
                                     log_path=self.outfilename+'/',
                                     eval_freq=int(1000),
                                     deterministic=True,
                                     render=False)
        
        self.model.learn(total_timesteps=int(5e7),
                         callback=eval_callback,
                         log_interval=100)

        # Save the model
        self.model.save(self.outfilename+'/final_model.zip')
        print(self.outfilename)

        # Print training progression
        with np.load(self.outfilename+'/evaluations.npz') as data:
            for j in range(data['timesteps'].shape[0]):
                print(str(data['timesteps'][j])+","+str(data['results'][j][0]))

if __name__ == "__main__":
    learn = LearnHoverS3()
    learn.train(RLatt)
