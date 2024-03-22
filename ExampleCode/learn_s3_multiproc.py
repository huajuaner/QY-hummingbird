import sys 
sys.path.append('D://graduate//fwmav//simul2024//240312//QY-hummingbird-main')
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
from cycler import cycler

from hitsz_qy_hummingbird.envs.RL_wrapped import RLMAV
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

DEFAULT_OUTPUT_FOLDER = 'results'

class learn_hover_s3():
    def __init__(self):
        self.output_folder=DEFAULT_OUTPUT_FOLDER
        self.num_mavs = 1
    
    def train(self):
        self.filename = os.path.join(self.output_folder, 'save-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
        if not os.path.exists(self.filename):
            os.makedirs(self.filename+'/')
    
        train_env = make_vec_env(RLMAV,
                                 n_envs=12,
                                 seed=0,
                                 vec_env_cls=SubprocVecEnv,
                                 )
        eval_env = make_vec_env(RLMAV,
                                 n_envs=12,
                                 seed=0,
                                 vec_env_cls=SubprocVecEnv,
                                 )

        #### Check the environment's spaces ########################
        print('[INFO] Action space:', train_env.action_space)
        print('[INFO] Observation space:', train_env.observation_space)

        #### Train the model #######################################
        model = PPO('MlpPolicy',
                    train_env,
                    # tensorboard_log=filename+'/tb/',
                    verbose=1)
        
        target_reward = 1e10
        callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=target_reward,
                                                        verbose=1)
        
        #EvalCallback 的主要作用是在训练期间定期评估模型，并在找到新的最佳模型时保存它。这对于监控模型的性能以及在训练过程中选择最佳模型是很有用的。
        #eval_freq: 评估的频率，表示每隔多少个训练步数执行一次评估。在这里，设置为 int(1000)，表示每隔1000个训练步执行一次评估。
        eval_callback = EvalCallback(eval_env,
                                    callback_on_new_best=callback_on_best,
                                    verbose=1,
                                    best_model_save_path=self.filename+'/',
                                    log_path=self.filename+'/',
                                    eval_freq=int(1000),
                                    deterministic=True,
                                    render=False)
        
        model.learn(total_timesteps=int(50000), 
                callback=eval_callback,
                log_interval=100)

        #### Save the model ########################################
        model.save(self.filename+'/final_model.zip')
        print(self.filename)

        #### Print training progression ############################
        with np.load(self.filename+'/evaluations.npz') as data:
            for j in range(data['timesteps'].shape[0]):
                print(str(data['timesteps'][j])+","+str(data['results'][j][0]))

if __name__ == "__main__":
    learn = learn_hover_s3()
    learn.train()

