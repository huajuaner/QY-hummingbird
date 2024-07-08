'''Training hht-RL models in serial, using the default DummyVecEnv in stablebaselines3.'''

import sys

sys.path.append('/home/hht/simul0703/QY-hummingbird/')
import os
from datetime import datetime

from hitsz_qy_hummingbird.envs.rl_attitude import RLatt

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

DEFAULT_OUTPUT_FOLDER = 'results'


class learn_hover_s3():
    def __init__(self):
        self.output_folder = DEFAULT_OUTPUT_FOLDER
        self.num_mavs = 1

    def train(self, my_env):
        self.filename = os.path.join(self.output_folder, 'save-' + datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
        if not os.path.exists(self.filename):
            os.makedirs(self.filename + '/')

        train_env = make_vec_env(my_env,
                                 n_envs=2,
                                 seed=0,
                                 )
        eval_env = make_vec_env(my_env,
                                n_envs=2,
                                seed=0,
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

        # The primary role of EvalCallback is to periodically evaluate the model during training and save it when a new best model is found.
        # eval_freq: Evaluation frequency, indicating how often evaluation is performed, measured in terms of the number of training steps.
        # Here, it is set to int(1000), meaning evaluation is performed every 1000 training steps.
        eval_callback = EvalCallback(eval_env,
                                     callback_on_new_best=callback_on_best,
                                     verbose=1,
                                     best_model_save_path=self.filename + '/',
                                     log_path=self.filename + '/',
                                     eval_freq=int(1000),
                                     deterministic=True,
                                     render=False)

        model.learn(total_timesteps=int(10000),
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
    learn = learn_hover_s3()
    learn.train(RLhover)
