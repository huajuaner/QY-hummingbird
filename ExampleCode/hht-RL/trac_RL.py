'''test the trained hht-RL model with GUI for trajectory tracking'''

import sys

sys.path.append('/home/hht/simul0703/QY-hummingbird/')
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.legend_handler import HandlerBase
from cycler import cycler

from hitsz_qy_hummingbird.wrapper.wrapped_mav_for_RL import RLMAV
from hitsz_qy_hummingbird.envs.rl_attitude import RLatt
from hitsz_qy_hummingbird.envs.rl_trac import RLtrac
from hitsz_qy_hummingbird.envs.rl_flip import RLflip
from hitsz_qy_hummingbird.envs.rl_escape import RLescape
from hitsz_qy_hummingbird.configuration import configuration
from hitsz_qy_hummingbird.base_FWMAV.wings.wing_params import ParamsForBaseWing

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

DEFAULT_OUTPUT_FOLDER = 'att_results'

# Define a custom handler for legend
class HandlerLine2DWithWidth(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        # Create a horizontal line with the appropriate width
        legline = Line2D([0, width], [height/2.0, height/2.0], linewidth=4, color=orig_handle.get_color())
        self.update_prop(legline, orig_handle, legend)
        legline.set_transform(trans)
        return [legline]

    def update_prop(self, legend_handle, orig_handle, legend):
        legend_handle.set_label(orig_handle.get_label())

class testRLtrac():
    def __init__(self):
        self.initial_pos=np.array([0.2,0,0.5])
        self.initial_att=np.array([0,0,0])
        self.target_pos=np.array([0.2,0,0.5])
        self.trac_pos=np.array([0.2,0,0.5])
        self.target_att=np.array([0,0,0])
        self.output_folder = DEFAULT_OUTPUT_FOLDER
        self.num_mavs = 1
        self.counters = np.zeros(self.num_mavs)
        self.timesteps = np.zeros((self.num_mavs, 0))
        self.states = np.zeros((self.num_mavs, 16, 0))
        self.otherdata = np.zeros((self.num_mavs, 4, 0))

    def test(self):

        path = DEFAULT_OUTPUT_FOLDER+'/save-05.06.2024_20.24.09/best_model.zip'
        
        model = PPO.load(path)

        test_env = RLtrac(gui=True,
                         trunc_flag=False,
                         initial_pos=self.initial_pos,
                         target_pos=self.target_pos,
                         initial_att=self.initial_att,
                         target_att=self.target_att,)
        os.system("wmctrl -r :ACTIVE: -b add,fullscreen")
        test_env.add_debug_circle_xy()
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
        print('i=0,obs:')
        obs, info = test_env.reset(self.trac_pos)
        print(obs[0:12])
        start = time.time()
        for i in range(2000):
            test_env.pause()
        whole_ctrl_steps = 10*int(test_env.CTRL_FREQ)
        #every up_pos ctrlsteps update trac_pos
        up_pos=600
        for i in range(whole_ctrl_steps):
            # for i in range((test_env.EPISODE_LEN_SEC+2)):
            action, _states = model.predict(obs,
                                            deterministic=True
                                            )
            if(((i+up_pos)%up_pos)==0):
                self.trac_pos = self.Circle_trac_xy((i+1.4*up_pos)/whole_ctrl_steps)
            obs, reward, terminated, truncated, info = test_env.step(action,trac_pos=self.trac_pos)
            if (i==1):
                print('i=1,obs:')
                print(obs[0:12])
            # if (i==300):
            #     test_env.mav.snapshot(DEFAULT_OUTPUT_FOLDER+'/test1.png')
            otherdata = test_env._getotherdata()
            true_xyzrpy = test_env._gettrue_xyzrpy()
            state = np.hstack([true_xyzrpy,obs[6:12],action])
            self.log_perstep(i, state, otherdata)
            if terminated or truncated:
                obs, info = test_env.reset(seed=7)
            
                
        test_env.close()

    def Circle_trac_xy(self,timefrac):
        R=0.2
        theta = timefrac * 2 * np.pi  # 角度
        x = -0.02+0.95*(R * np.cos(theta)-2*R* np.sin(theta))  # x 坐标
        y = 1.3*(R * np.sin(theta)+2*R* np.cos(theta)) # y 坐标
        z = self.initial_pos[2]  # z 坐标设为初始值
        return np.array([x, y, z])
    
    def Circle_trac_xz(self,timefrac):
        R=0.2
        theta = timefrac * 2 * np.pi+0.5*np.pi  # 角度
        # x = 0.9*(R * np.cos(theta)-2*R * np.sin(theta)) # x 坐标
        # y = self.initial_pos[1] # y 坐标
        # z = 0.3+0.9*(R * np.sin(theta)+2*R * np.cos(theta)) # z 坐标
        x = 2.5*R * np.cos(theta)  # x 坐标
        y = self.initial_pos[1] # y 坐标
        z = 0.3+R * np.sin(theta) # z 坐标
        return np.array([x, y, z])

    def log_perstep(self, timestep, state, otherdata, drone=0):
        # log
        # current drone's counter
        current_counter = int(self.counters[drone])
        #
        if current_counter >= self.timesteps.shape[1]:
            self.timesteps = np.concatenate((self.timesteps, np.zeros((self.num_mavs, 1))), axis=1)
            self.states = np.concatenate((self.states, np.zeros((self.num_mavs, 16, 1))), axis=2)
            self.otherdata = np.concatenate((self.otherdata, np.zeros((self.num_mavs, 4, 1))), axis=2)
        elif current_counter < self.timesteps.shape[1]:
            current_counter = self.timesteps.shape[1] - 1
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

        #### XYZ #############################################_gettrue_xyzrpy
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('x_e (m)')

        row = 1
        for j in range(self.num_mavs):
            axs[row, col].plot(t, self.states[j, 1, :], )
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('y_e (m)')

        row = 2
        for j in range(self.num_mavs):
            axs[row, col].plot(t,  self.states[j, 2, :], )
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('z_e (m)')

        #### RPY ###################################################
        row = 3
        for j in range(self.num_mavs):
            axs[row, col].plot(t, np.pi * self.states[j, 3, :], )
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('r_e (rad)')
        row = 4
        for j in range(self.num_mavs):
            axs[row, col].plot(t, (np.pi / 2) * self.states[j, 4, :], )
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('p_e (rad)')
        row = 5
        for j in range(self.num_mavs):
            axs[row, col].plot(t, np.pi * self.states[j, 5, :], )
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('y_e (rad)')

        #### vx  vy ###################################################
        row = 6
        for j in range(self.num_mavs):
            axs[row, col].plot(t, self.states[j, 6, :], )
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('vx (m/s)')

        row = 7
        for j in range(self.num_mavs):
            axs[row, col].plot(t, self.states[j, 7, :], )
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('vy (m/s)')

        row = 8
        for j in range(self.num_mavs):
            axs[row, col].plot(t, self.otherdata[j, 2, :], )
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('r_stroke (rad)')

        row = 9
        for j in range(self.num_mavs):
            axs[row, col].plot(t, self.otherdata[j, 3, :], )
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('l_stroke (rad)')

        #### Column ################################################
        col = 1

        #### vz ###################################################
        row = 0
        for j in range(self.num_mavs):
            axs[row, col].plot(t, self.states[j, 8, :], )
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('vz (m/s)')

        #### wx wy wz###################################################
        row = 1
        for j in range(self.num_mavs):
            axs[row, col].plot(t, self.states[j, 9, :], )
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('wx (rad/s)')
        row = 2
        for j in range(self.num_mavs):
            axs[row, col].plot(t, self.states[j, 10, :], )
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('wy (rad/s)')
        row = 3
        for j in range(self.num_mavs):
            axs[row, col].plot(t, self.states[j, 11, :], )
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('wz (rad/s)')

        #### U dU U0 sc###################################################
        row = 4
        for j in range(self.num_mavs):
            axs[row, col].plot(t, self.states[j, 12, :], )
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('U (V)')

        row = 5
        for j in range(self.num_mavs):
            axs[row, col].plot(t, self.states[j, 13, :], )
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('dU (V)')

        row = 6
        for j in range(self.num_mavs):
            axs[row, col].plot(t, self.states[j, 14, :], )
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('U0 (V)')

        row = 7
        for j in range(self.num_mavs):
            axs[row, col].plot(t, self.states[j, 15, :], )
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('sc')

        row = 8
        for j in range(self.num_mavs):
            axs[row, col].plot(t, self.otherdata[j, 0, :], )
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('r_u (V)')

        row = 9
        for j in range(self.num_mavs):
            axs[row, col].plot(t, self.otherdata[j, 1, :], )
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('l_u (V)')
        
        fig.subplots_adjust(left=0.06,
                            bottom=0.05,
                            right=0.99,
                            top=0.98,
                            wspace=0.15,
                            hspace=0.0
                            )
        plt.show()

    def plot_xyzrpy(self):
        #### Loop over colors and line styles ######################
        # plt.rc('axes', prop_cycle=(cycler('color', ['r', 'y', 'b']) ))
        plt.rc('axes', prop_cycle=(cycler('color', ['b', 'g', 'r', 'y']) ))
        fig, axs = plt.subplots(6, 1)
        # t = np.arange(0, self.timesteps.shape[1]/GLOBAL_CONFIGURATION.TIMESTEP-1/GLOBAL_CONFIGURATION.TIMESTEP, 1/GLOBAL_CONFIGURATION.TIMESTEP)
        t = np.arange(0, self.timesteps.shape[1])
        alp_size=20
        tick_size = 20
        #### Column ################################################

        #### XYZ ###################################################
        row = 0
        for j in range(self.num_mavs):
            axs[row].plot(t,  self.states[j, 0, :], label="mav_" + str(j))
        axs[row].set_xlabel('control step',loc='right', fontsize=alp_size)
        axs[row].set_ylabel('x (m)', fontsize=alp_size)
        axs[row].tick_params(axis='x', labelsize=tick_size )
        axs[row].tick_params(axis='y', labelsize=tick_size )

        row = 1
        for j in range(self.num_mavs):
            axs[row].plot(t, self.states[j, 1, :] )
        axs[row].set_xlabel('control step',loc='right', fontsize=alp_size)
        axs[row].set_ylabel('y (m)', fontsize=alp_size)
        axs[row].tick_params(axis='x', labelsize=tick_size )
        axs[row].tick_params(axis='y', labelsize=tick_size )

        row = 2
        for j in range(self.num_mavs):
            axs[row].plot(t,  self.states[j, 2, :], label="mav_" + str(j))
        axs[row].set_xlabel('control step',loc='right', fontsize=alp_size)
        axs[row].set_ylabel('z (m)', fontsize=alp_size)
        axs[row].tick_params(axis='x', labelsize=tick_size )
        axs[row].tick_params(axis='y', labelsize=tick_size )
        # axs[row].set_yticks([0,0.5,1,1.5,2,2.5])


        #### RPY ###################################################
        row = 3
        for j in range(self.num_mavs):
            axs[row].plot(t, (180/np.pi)*self.states[j, 1, :], label="$U_a=9V$" )
        axs[row].set_xlabel('control step',loc='right', fontsize=alp_size)
        axs[row].set_ylabel('roll angle (°)', fontsize=alp_size)
        axs[row].tick_params(axis='x', labelsize=tick_size )
        axs[row].tick_params(axis='y', labelsize=tick_size )

        row = 4
        for j in range(self.num_mavs):
            axs[row].plot(t,  (180/np.pi)*self.states[j, 4, :], label="mav_" + str(j))
        axs[row].set_xlabel('control step',loc='right', fontsize=alp_size)
        axs[row].set_ylabel('pitch angle (°)', fontsize=alp_size)
        axs[row].tick_params(axis='x', labelsize=tick_size )
        axs[row].tick_params(axis='y', labelsize=tick_size )

        row = 5
        for j in range(self.num_mavs):
            axs[row].plot(t, (180/np.pi)*self.states[j, 5, :], label="mav_" + str(j))
        axs[row].set_xlabel('control step',loc='right', fontsize=alp_size)
        axs[row].set_ylabel('yaw angle (°)', fontsize=alp_size)
        axs[row].tick_params(axis='x', labelsize=tick_size )
        axs[row].tick_params(axis='y', labelsize=tick_size )

        
        # fig.subplots_adjust(left=0.06,
        #                     bottom=0.05,
        #                     right=0.99,
        #                     top=0.98,
        #                     wspace=0.15,
        #                     hspace=0.0
        #                     )
        # axs[0,1].legend(fontsize=22, loc='upper right')
        fig.subplots_adjust(left=0.125, right=0.879,top=0.983,bottom=0.09, wspace=0.271,hspace=0.514)
        plt.show()

    def plot_xyz(self):
        fig, axs = plt.subplots()
        axs2 = axs.twinx()
        axs3 = axs.twinx()
        t = np.arange(0, self.timesteps.shape[1])
        t_real = t/1200
        alp_size = 20
        tick_size = 20

        handles = []  # List to hold handles for legend
        labels = []   # List to hold labels for legend

        for j in range(self.num_mavs):
            # Plot x data on first axis
            handle, = axs.plot(t_real, self.states[j, 0, :]*1e3, linewidth=3, label="x", color='tab:red')
            handles.append(handle)
            labels.append("x")

            # Plot y data on second axis
            handle, = axs2.plot(t_real, self.states[j, 1, :]*1e3, linewidth=3, label="y", color='tab:green')
            handles.append(handle)
            labels.append("y")

            # Plot z data on third axis
            handle, = axs3.plot(t_real, self.states[j, 2, :]*1e3, linewidth=3, label="z", color='tab:blue')
            handles.append(handle)
            labels.append("z")

        # axs.set_xlabel('control step', loc='right', fontsize=alp_size)
        axs.set_xlabel('time(s)', loc='right', fontsize=alp_size)  
        axs.set_ylabel('position (mm)', fontsize=alp_size)
        axs.set_xticks(np.arange(0, 11,1))
        axs.set_yticks(np.arange(-300, 700, 100))
        axs2.set_yticks(np.arange(-300, 700, 100))
        axs3.set_yticks(np.arange(-300, 700, 100))
        axs.tick_params(axis='x', labelsize=tick_size)
        axs.tick_params(axis='y', labelsize=tick_size)
        axs2.tick_params(axis='y', labelsize=tick_size)
        axs3.tick_params(axis='y', labelsize=tick_size)

        axs.legend(handles, labels, loc='upper right', fontsize=alp_size)
        fig.subplots_adjust(left=0.202, right=0.9,top=0.88,bottom=0.11, wspace=0.2,hspace=0.2)
        plt.show()

    def plot_rpy(self):
        fig, axs = plt.subplots()
        axs2 = axs.twinx()
        axs3 = axs.twinx()
        t = np.arange(0, self.timesteps.shape[1])
        t_real = t/1200
        alp_size = 20
        tick_size = 20

        handles = []  # List to hold handles for legend
        labels = []   # List to hold labels for legend

        j=0

        #注意这里的顺序是pyr
        handle, = axs.plot(t_real, (180/np.pi)*(self.target_att[1]+self.states[j, 4, :]), linewidth=0.5, label="pitch", color='tab:green')
        handles.append(handle)
        labels.append("pitch")

        handle, = axs2.plot(t_real, (180/np.pi)*(self.target_att[2]+self.states[j, 5, :]), linewidth=0.5, label="yaw", color='tab:blue')
        handles.append(handle)
        labels.append("yaw")

        handle, = axs3.plot(t_real, (180/np.pi)*(self.target_att[0]+self.states[j, 3, :]), linewidth=1, label="roll", color='tab:red')
        handles.append(handle)
        labels.append("roll")
        # 将 roll 的 handle 和 label 插入到列表的开头
        handles.insert(0, handle)
        labels.insert(0, "roll")
        handles.pop()
        labels.pop()

        axs.set_xlabel('time(s)', loc='right', fontsize=alp_size)
        axs.set_ylabel('attitude (deg)', fontsize=alp_size)
        axs.set_xticks(np.arange(0, 11,1))
        axs.set_yticks(np.arange(-12, 10, 2))
        axs2.set_yticks(np.arange(-12, 10, 2))
        axs3.set_yticks(np.arange(-12, 10, 2))
        axs.tick_params(axis='x', labelsize=tick_size)
        axs.tick_params(axis='y', labelsize=tick_size)
        axs2.tick_params(axis='y', labelsize=tick_size)
        axs3.tick_params(axis='y', labelsize=tick_size)

        # Use the custom handler for legend
        axs.legend(handles, labels, loc='upper right', fontsize=16, handler_map={Line2D: HandlerLine2DWithWidth()})
        fig.subplots_adjust(left=0.202, right=0.9,top=0.88,bottom=0.11, wspace=0.2,hspace=0.2)
        plt.show()

if __name__ == "__main__":
    learn = testRLtrac()
    learn.test()
    learn.plot_xyz()
    learn.plot_rpy()
