'''Training hht-RL models in parallel, using the SubprocVecEnv in stablebaselines3.'''

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


class learn_hover_s3():
    def __init__(self):
        self.output_folder = DEFAULT_OUTPUT_FOLDER
        self.num_mavs = 1

    def train(self, my_env):
        self.filename = os.path.join(self.output_folder, 'save-' + datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
        if not os.path.exists(self.filename):
            os.makedirs(self.filename + '/')

        train_env = make_vec_env(my_env,
                                 n_envs=20,
                                 seed=0,
                                 vec_env_cls=SubprocVecEnv,
                                 )
        eval_env = make_vec_env(my_env,
                                n_envs=20,
                                seed=0,
                                vec_env_cls=SubprocVecEnv,
                                )

        #### Check the environment's spaces ########################
        print('[INFO] Action space:', train_env.action_space)
        print('[INFO] Observation space:', train_env.observation_space)

        #### Train the model #######################################
        model = PPO('MlpPolicy',
                    train_env,
                    batch_size=256, 
                    gamma=0.99, 
                    device="cuda",
                    # tensorboard_log=filename+'/tb/',
                    verbose=1)

        target_reward = 1e10
        callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=target_reward,
                                                         verbose=1)

        eval_callback = EvalCallback(eval_env,
                                     callback_on_new_best=callback_on_best,
                                     verbose=1,
                                     best_model_save_path=self.filename + '/',
                                     log_path=self.filename + '/',
                                     eval_freq=int(1000),
                                     deterministic=True,
                                     render=False)

        model.learn(total_timesteps=int(2e5),
                    callback=eval_callback,
                    log_interval=100)

        #### Save the model ########################################
        model.save(self.filename + '/final_model.zip')
        print(self.filename)

        #### Print training progression ############################
        with np.load(self.filename + '/evaluations.npz') as data:
            for j in range(data['timesteps'].shape[0]):
                print(str(data['timesteps'][j]) + "," + str(data['results'][j][0]))


if __name__ == "__main__":

    configuration.ParamsForMAV_rl.change_parameters(sleep_time=0)
    learn = learn_hover_s3()
    learn.train(RLatt)
