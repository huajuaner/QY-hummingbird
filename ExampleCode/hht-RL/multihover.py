'''
Compare to hover.py, this file 
May be comblined into a single file in the future, yet this file call the p.stepSimulation() explicitly.
This file controls self.num_mavs FWMAVs, and the idea of controlling two FWMAVs is as follows:
# def two_envs():
#     test_env1 = RLatt(gui=True,pyb=p,client=physicsCilent,initial_pos=([0,0,0.5]),target_pos=np.array([0,0,0.5]))
#     test_env2 = RLatt(gui=True,pyb=p,client=physicsCilent,initial_pos=([0,0,1.0]),target_pos=np.array([0,0,1.0]),)

#     obs1, info1 = test_env1.reset()
#     obs2, info2 = test_env2.reset()
#     print(obsfrom hitsz_qy_hummingbird.envs.rl_hover import RLhover1,obs2)
#     for i in range(10*int(test_env1.CTRL_FREQ)):
#         action1, _states = model.predict(obs1,
#                                     deterministic=True
#                                     )
#         action2, _states = model.predict(obs2,
#                                     deterministic=True
#                                     )
#         for _ in range(20): 
#             test_env1.prestep(action1)
#             test_env2.prestep(action2)
#             p.stepSimulation()
#             time.sleep(sleep_time)
#             test_env1.mav.joint_state_update()
#             test_env2.mav.joint_state_update()
#         obs1, reward1, terminated, truncated, info = test_env1.step_after_pyb(action1)
#         obs2, reward2, terminated, truncated, info = test_env2.step_after_pyb(action2)

#     p.disconnect()
'''


import sys
sys.path.append('/home/hht/simul0703/QY-hummingbird/')
import os
import time
from cycler import cycler
import math


from hitsz_qy_hummingbird.envs.rl_attitude import RLatt
from hitsz_qy_hummingbird.envs.rl_flip import RLflip
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import pybullet as p
import time
import pybullet_data
from hitsz_qy_hummingbird.configuration.configuration import GLOBAL_CONFIGURATION

PYB_STEPS_PER_CTRL=24000//1200
DEFAULT_OUTPUT_FOLDER = 'att_results'
class Hover():
    def __init__(self):
        self.set_points = Twins()
        self.difatt_flag = False
        self.set_att = CircleAtt()
        self.num_mavs = self.set_points.shape[0]
        self.counters = np.zeros(self.num_mavs)
        self.timesteps = np.zeros((self.num_mavs, 0))
        self.states = np.zeros((self.num_mavs, 16, 0))
        self.otherdata = np.zeros((self.num_mavs, 4, 0))
        
    def hover(self):
        sleep_time=0.0001

        physicsCilent = p.connect(p.GUI)
        os.system("wmctrl -r :ACTIVE: -b add,fullscreen")
        # print(f"physicsCilent is {physicsCilent}")
        lightPosition = [10, -10, 20]
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        p.configureDebugVisualizer(lightPosition=lightPosition)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-9.8)
        p.loadURDF("plane.urdf")
        p.setTimeStep(1 / GLOBAL_CONFIGURATION.TIMESTEP)

        path = DEFAULT_OUTPUT_FOLDER+'/save-05.06.2024_20.24.09/best_model.zip'
        model = PPO.load(path)

        envs = []
        observations = []
        actions = []
        infos = []

        for i in range(self.num_mavs):
            if  self.difatt_flag==True:
                test_env = RLatt(gui=True,
                                trunc_flag=False,
                                pyb=p,
                                client=physicsCilent,
                                initial_pos=self.set_points[i]-np.array([0,0,0.5]),
                                target_pos=self.set_points[i],
                                initial_att=self.set_att[i],
                                target_att=self.set_att[i])
            else:
                test_env = RLatt(gui=True,
                                trunc_flag=False,
                                pyb=p,
                                client=physicsCilent,
                                initial_pos=self.set_points[i],
                                target_pos=self.set_points[i])
            envs.append(test_env)

        for i in range(self.num_mavs):
            obs, info = envs[i].reset_keep_env()
            observations.append(obs)
            # otherdata = envs[i]._getotherdata()
            # self.log_perstep(0,obs,otherdata,i)
            infos.append(info)
            actions.append(np.zeros(4))

        m=20
        k=0
        # ct=control step
        # m and k are parameters used to control the sequential entry of multiple aircraft into the control mode
        # m is a fixed value. 
        # k increments by 1 at each control step, and an aircraft enters control every m control-steps starting from 0ct.
        # i=mavs[i]
        for ct in range(int(5*envs[0].CTRL_FREQ)):
            # if ct==int(0.1*envs[0].CTRL_FREQ):
            #     time.sleep(5)
            # if ct==int(5*envs[0].CTRL_FREQ):
            #     time.sleep(10)
            for i in range(self.num_mavs):
                actions[i], _states = model.predict(observations[i],
                                            deterministic=True
                                            )

            for _ in range(PYB_STEPS_PER_CTRL): 
                for i in range(self.num_mavs):
                    envs[i].prestep(actions[i],((m*i-k)>0))
                p.stepSimulation()
                time.sleep(sleep_time)
                for i in range(self.num_mavs):
                    envs[i].mav.joint_state_update()

            for i in range(self.num_mavs):
                observations[i], reward, terminated, truncated, info = envs[i].step_after_pyb(actions[i])
                otherdata = envs[i]._getotherdata()
                true_xyzrpy =envs[i]._gettrue_xyzrpy()
                state = np.hstack([true_xyzrpy[0:6],observations[i][6:12],actions[i]])
                self.log_perstep(ct, state, otherdata,i)
            k=k+1

        p.disconnect()

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
    
    def plot_z(self):
        #### Loop over colors and line styles ######################
        # plt.rc('axes', prop_cycle=(cycler('color', ['r', 'y', 'b']) ))
        colors = ['b', 'g', 'r', 'y', 'c', 'm', 'k', 'orange']
        plt.rc('axes', prop_cycle=(cycler('color', colors)))
        fig, axs = plt.subplots()
        t = np.arange(0, self.timesteps.shape[1])
        t_real = t/1200
        alp_size = 20
        tick_size = 20
        for j in range(self.num_mavs):
            axs.plot(t_real,  self.states[j, 2, :]*1000, label="fwmav_" + str(j))
        # axs.set_xlabel('control step',loc='right', fontsize=alp_size)  
        axs.set_xlabel('time(s)',loc='right', fontsize=alp_size)          
        axs.set_ylabel('z (mm)', fontsize=alp_size)
        axs.tick_params(axis='x', labelsize=tick_size )
        axs.tick_params(axis='y', labelsize=tick_size )
        plt.legend(loc='lower right', fontsize=alp_size)
        plt.show()

def Twins():
    return np.array([[0,0,0.5],[0,-0.3,0.5]])

def CubePoints():
    return np.array([[0,0,0.3],[-0.3,-0.3,0.3],[-0.3,0,0.3],[0,-0.3,0.3],
                       [0,0,0.6],[-0.3,-0.3,0.6],[-0.3,0,0.6],[0,-0.3,0.6]])

def CirclePoints():
    # Define the coordinates of the four vertices of the square
    square_points = [[0, 0, 0.75], [-0.5, -0.5, 0.75], [-0.5, 0, 0.75], [0, -0.5, 0.75]]
    # Calculate the coordinates of the center of the square
    center = [sum(p[0] for p in square_points) / 4,
              sum(p[1] for p in square_points) / 4,
              sum(p[2] for p in square_points) / 4]
    # Calculate the radius of the inscribed circle
    radius = 0.5 / math.sqrt(2)
    # Generate 8 equally spaced points
    circle_points = []
    for i in range(8):
        angle = i * (2 * math.pi / 8)
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        z = center[2]
        circle_points.append([x, y, z])
    return np.array(circle_points)

def CircleAtt():
    circleatt = [[0,0,0],[0,0,np.pi/4],[0,0,np.pi/2],[0,0,3*np.pi/4],[0,0,np.pi],
                 [0,0,-3*np.pi/4],[0,0,-np.pi/2],[0,0,-np.pi/4],]
    return np.array(circleatt)


if __name__ == "__main__":
    test=Hover()
    test.hover()
    test.plot_z()
