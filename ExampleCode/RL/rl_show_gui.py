'''test the trained RL model with GUI'''

import sys 
sys.path.append('D://graduate//fwmav//simul2024//240325git//QY-hummingbird')
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
from cycler import cycler

from hitsz_qy_hummingbird.envs.RL_wrapped import RLMAV
from hitsz_qy_hummingbird.envs.rl_hover import RLhover
from hitsz_qy_hummingbird.configuration import configuration
from hitsz_qy_hummingbird.base_FWMAV.wings.wing_params import ParamsForBaseWing
from hitsz_qy_hummingbird.configuration.configuration import GLOBAL_CONFIGURATION

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy

DEFAULT_OUTPUT_FOLDER = 'results'

class testRL():
    def __init__(self):
        self.output_folder=DEFAULT_OUTPUT_FOLDER
        self.num_mavs = 1
        self.counters = np.zeros(self.num_mavs)
        self.timesteps = np.zeros((self.num_mavs, 0))
        self.states = np.zeros((self.num_mavs, 16, 0))
        self.otherdata = np.zeros((self.num_mavs, 4, 0))

    def test(self):

        path = 'D://graduate//fwmav//simul2024/240315show//QY-hummingbird-main//ExampleCode//result_remote//save-03.22.2024_23.44.20//best_model.zip'
        # path = 'D://graduate//fwmav//simul2024//240201//results//save-03.10.2024_07.41.58//best_model.zip'
        model = PPO.load(path)

        test_env = RLhover(gui=True)

        # Using the evaluate_policy function from the Stable Baselines3 library to evaluate the performance of the trained reinforcement learning model on the test environment.
        # Specifically, it calculates the average reward and the standard deviation of rewards over multiple evaluation cycles in the test environment.
        # Parameters:
        # model: The trained reinforcement learning model.
        # test_env: The test environment, which may be an independent instance of the environment used to assess model performance.
        # n_eval_episodes: The number of evaluation cycles, i.e., the number of times to evaluate the model's performance.
        # The function returns two values:
        # mean_reward: The average reward obtained over multiple evaluations in the test environment.
        # std_reward: The standard deviation of rewards obtained over multiple evaluations in the test environment.
        
        # mean_reward, std_reward = evaluate_policy(model,
        #                                         test_env,
        #                                         n_eval_episodes=10
        #                                         )
        # print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")

        obs, info = test_env.reset()
        print(obs)
        # start = time.time()
        for i in range(int((test_env.EPISODE_LEN_SEC+2)*test_env.CTRL_FREQ/3)):
        #for i in range((test_env.EPISODE_LEN_SEC+2)):
            action, _states = model.predict(obs,
                                        deterministic=True
                                        )
            obs, reward, terminated, truncated, info = test_env.step(action)
            otherdata = test_env._getotherdata()
            self.log(i, obs, otherdata)
            if terminated or truncated:
                obs, info  = test_env.reset(seed=7)
        test_env.close()

    def log(self,timestep,state, otherdata, drone =0):    
        #log     
        #current drone's counter
        current_counter = int(self.counters[drone])
        #
        if current_counter >= self.timesteps.shape[1]:
            self.timesteps = np.concatenate((self.timesteps, np.zeros((self.num_mavs, 1))), axis=1)
            self.states = np.concatenate((self.states, np.zeros((self.num_mavs, 16, 1))), axis=2)
            self.otherdata = np.concatenate((self.otherdata , np.zeros((self.num_mavs, 4, 1))), axis=2)
        elif  current_counter < self.timesteps.shape[1] :
            current_counter = self.timesteps.shape[1]-1
        self.timesteps[drone, current_counter] = timestep

        self.states[drone, :, current_counter] = state
        self.otherdata[drone, :, current_counter] = otherdata
        self.counters[drone] = current_counter + 1

    def plot(self):
        #### Loop over colors and line styles ######################
        plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y']) + cycler('linestyle', ['-', '--', ':', '-.'])))
        fig, axs = plt.subplots(10, 2)
        # t = np.arange(0, self.timesteps.shape[1]/GLOBAL_CONFIGURATION.TIMESTEP-1/GLOBAL_CONFIGURATION.TIMESTEP, 1/GLOBAL_CONFIGURATION.TIMESTEP)
        t = np.arange(0, self.timesteps.shape[1])
        #### Column ################################################
        col = 0

        #### XYZ ###################################################
        row = 0
        for j in range(self.num_mavs):
            axs[row, col].plot(t, 2*self.states[j, 0, :], label="mav_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('x (m)')

        row = 1
        for j in range(self.num_mavs):
            axs[row, col].plot(t, 2*self.states[j, 1, :], label="mav_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('y (m)')

        row = 2
        for j in range(self.num_mavs):
            axs[row, col].plot(t, 2*self.states[j, 2, :], label="mav_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('z (m)')

        #### RPY ###################################################
        row = 3
        for j in range(self.num_mavs):
            axs[row, col].plot(t, np.pi * self.states[j, 3, :], label="mav_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('r (rad)')
        row = 4
        for j in range(self.num_mavs):
            axs[row, col].plot(t, (np.pi/2) * self.states[j, 4, :], label="mav_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('p (rad)')
        row = 5
        for j in range(self.num_mavs):
            axs[row, col].plot(t, np.pi * self.states[j, 5, :], label="mav_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('y (rad)')

        #### vx  vy ###################################################
        row = 6
        for j in range(self.num_mavs):
            axs[row, col].plot(t, self.states[j, 6, :], label="mav_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('vx (m/s)')

        row = 7
        for j in range(self.num_mavs):
            axs[row, col].plot(t, self.states[j, 7, :], label="mav_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('vy (m/s)')

        row = 8
        for j in range(self.num_mavs):
            axs[row, col].plot(t, self.otherdata[j, 2, :], label="mav_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('right_stroke_amp (rad)')

        row = 9
        for j in range(self.num_mavs):
            axs[row, col].plot(t, self.otherdata[j, 3, :], label="mav_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('left_stroke_amp (rad)')

        #### Column ################################################
        col = 1

        #### vz ###################################################
        row = 0
        for j in range(self.num_mavs):
            axs[row, col].plot(t, self.states[j, 8, :], label="mav_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('vz (m/s)')

        #### wx wy wz###################################################
        row = 1
        for j in range(self.num_mavs):
            axs[row, col].plot(t, self.states[j, 9, :], label="mav_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('wx (rad/s)')
        row = 2
        for j in range(self.num_mavs):
            axs[row, col].plot(t, self.states[j, 10, :], label="mav_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('wy (rad/s)')
        row = 3
        for j in range(self.num_mavs):
            axs[row, col].plot(t, self.states[j, 11, :], label="mav_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('wz (rad/s)')

        #### U dU U0 sc###################################################
        row = 4
        for j in range(self.num_mavs):
            axs[row, col].plot(t, self.states[j, 12, :], label="mav_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('U (V)')  

        row = 5
        for j in range(self.num_mavs):
            axs[row, col].plot(t, self.states[j, 13, :], label="mav_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('dU (V)')  

        row = 6
        for j in range(self.num_mavs):
            axs[row, col].plot(t, self.states[j, 14, :], label="mav_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('U0 (V)')  

        row = 7
        for j in range(self.num_mavs):
            axs[row, col].plot(t, self.states[j, 15, :], label="mav_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('sc')  

        row = 8
        for j in range(self.num_mavs):
            axs[row, col].plot(t, self.otherdata[j, 0, :], label="mav_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('r_u (V)')

        row = 9
        for j in range(self.num_mavs):
            axs[row, col].plot(t, self.otherdata[j, 1, :], label="mav_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('l_u (V)')

        # Drawing options 
        for i in range (8):
            for j in range (2):
                axs[i, j].grid(True)
                axs[i, j].legend(loc='upper right',
                         frameon=True
                         )
        fig.subplots_adjust(left=0.06,
                            bottom=0.05,
                            right=0.99,
                            top=0.98,
                            wspace=0.15,
                            hspace=0.0
                            )
        plt.show()

if __name__ == "__main__":
    learn = testRL()
    learn.test()
    learn.plot()