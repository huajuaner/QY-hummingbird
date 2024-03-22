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

    def test(self):

        # if os.path.isfile(self.filename+'/best_model.zip'):
        #     path = self.filename+'/best_model.zip'
        # else:
        #     print("[ERROR]: no model under the specified path", self.filename)


        path = 'D://graduate//fwmav//simul2024//240201//QY-hummingbird-main//ExampleCode//result_remote//save-03.11.2024_00.38.16//best_model.zip'
        # path = 'D://graduate//fwmav//simul2024//240201//results//save-03.10.2024_07.41.58//best_model.zip'
        model = PPO.load(path)

        # if not multiagent:
        #     print(gui)
        #     test_env = HoverAviary(gui=gui,
        #                         obs=DEFAULT_OBS,
        #                         act=DEFAULT_ACT,
        #                         record=record_video)
        #     test_env_nogui = HoverAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT)
        # else:
        #     test_env = MultiHoverAviary(gui=gui,
        #                                     num_drones=DEFAULT_AGENTS,
        #                                     obs=DEFAULT_OBS,
        #                                     act=DEFAULT_ACT,
        #                                     record=record_video)
        test_env = RLMAV(gui=True)

        #使用 Stable Baselines3 库中的 evaluate_policy 函数来评估训练好的强化学习模型在测试环境上的性能。
        #具体来说，它计算了模型在测试环境中多个评估周期内的平均奖励和奖励的标准差。
        #参数：
        #model：已经训练好的强化学习模型。
        #test_env：测试环境，可能是用于评估模型性能的独立环境实例。
        #n_eval_episodes：评估周期的数量，即评估模型性能的次数。
        #函数返回两个值：
        #mean_reward：在测试环境中多次评估的平均奖励值。
        #std_reward：在测试环境中多次评估的奖励值的标准差。
        
        # mean_reward, std_reward = evaluate_policy(model,
        #                                         test_env,
        #                                         n_eval_episodes=10
        #                                         )
        # print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")

        obs, info = test_env.reset()
        print(obs)
        # start = time.time()
        for i in range((test_env.EPISODE_LEN_SEC+2)*test_env.CTRL_FREQ):
        #for i in range((test_env.EPISODE_LEN_SEC+2)):
            action, _states = model.predict(obs,
                                        deterministic=True
                                        )
            obs, reward, terminated, truncated, info = test_env.step(action)
            self.log(i,obs)
            if terminated or truncated:
                obs, info  = test_env.reset(seed=7)
        test_env.close()

    def log(self,timestep,state, drone =0):    
        #log     
        #当前drone的counter
        current_counter = int(self.counters[drone])
        #
        if current_counter >= self.timesteps.shape[1]:
            self.timesteps = np.concatenate((self.timesteps, np.zeros((self.num_mavs, 1))), axis=1)
            self.states = np.concatenate((self.states, np.zeros((self.num_mavs, 16, 1))), axis=2)
        elif  current_counter < self.timesteps.shape[1] :
            current_counter = self.timesteps.shape[1]-1
        self.timesteps[drone, current_counter] = timestep
        #### Re-order the kinematic obs (of most Aviaries) #########
        self.states[drone, :, current_counter] = state
        self.counters[drone] = current_counter + 1

    def plot(self):
        #### Loop over colors and line styles ######################
        plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y']) + cycler('linestyle', ['-', '--', ':', '-.'])))
        fig, axs = plt.subplots(8, 2)
        # t = np.arange(0, self.timesteps.shape[1]/GLOBAL_CONFIGURATION.TIMESTEP-1/GLOBAL_CONFIGURATION.TIMESTEP, 1/GLOBAL_CONFIGURATION.TIMESTEP)
        t = np.arange(0, self.timesteps.shape[1])
        #### Column ################################################
        col = 0

        #### XYZ ###################################################
        row = 0
        for j in range(self.num_mavs):
            axs[row, col].plot(t, self.states[j, 0, :], label="mav_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('x (m)')

        row = 1
        for j in range(self.num_mavs):
            axs[row, col].plot(t, self.states[j, 1, :], label="mav_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('y (m)')

        row = 2
        for j in range(self.num_mavs):
            axs[row, col].plot(t, self.states[j, 2, :], label="mav_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('z (m)')

        #### RPY ###################################################
        row = 3
        for j in range(self.num_mavs):
            axs[row, col].plot(t, self.states[j, 3, :], label="mav_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('r (rad)')
        row = 4
        for j in range(self.num_mavs):
            axs[row, col].plot(t, self.states[j, 4, :], label="mav_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('p (rad)')
        row = 5
        for j in range(self.num_mavs):
            axs[row, col].plot(t, self.states[j, 5, :], label="mav_"+str(j))
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

        #### Drawing options #######################################
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