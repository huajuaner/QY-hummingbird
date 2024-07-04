'''this file test the free-loop flight with no control signal'''
import os
import sys

sys.path.append('/home/hht/simul0703/240414/QY-hummingbird')


from hitsz_qy_hummingbird.wrapper.wrapped_mav_for_PID import WrappedMAVpid
from hitsz_qy_hummingbird.base_FWMAV.MAV.base_MAV_sequential import BaseMavSequential
from hitsz_qy_hummingbird.configuration import configuration
from hitsz_qy_hummingbird.base_FWMAV.motor.motor_params import ParamsForBaseMotor
from hitsz_qy_hummingbird.base_FWMAV.wings.wing_params import ParamsForBaseWing
from hitsz_qy_hummingbird.pid_controller.pidlinear import PIDLinear
from hitsz_qy_hummingbird.configuration.configuration import GLOBAL_CONFIGURATION
from hitsz_qy_hummingbird.utils.create_urdf_2 import URDFCreator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from cycler import cycler
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.legend_handler import HandlerBase

class HandlerLine2DWithWidth(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        # Create a horizontal line with the appropriate width
        legline = Line2D([0, width], [height/2.0, height/2.0], linewidth=4, color=orig_handle.get_color())
        self.update_prop(legline, orig_handle, legend)
        legline.set_transform(trans)
        return [legline]

    def update_prop(self, legend_handle, orig_handle, legend):
        legend_handle.set_label(orig_handle.get_label())

class free_loop():
    def __init__(self):

        self.num_mavs = 2
        self.counters = np.zeros(self.num_mavs)
        self.timesteps = np.zeros((self.num_mavs, 0))
        self.states = np.zeros((self.num_mavs, 16, 0))
        self.otherdata = np.zeros((self.num_mavs, 4, 0))

        self.model = PIDLinear()
        self.durtime = 0

        ng =10
        ar =5.2
        tr =0.8
        r22 =4e-6
        self.motor_params = configuration.ParamsForMaxonSpeed6M
        self.motor_params.change_parameters(spring_wire_diameter=0.7,
                                    spring_number_of_coils=6,
                                    spring_outer_diameter=3.5,
                                    gear_efficiency=0.8,
                                    gear_ratio=ng)

        self.wing_params = ParamsForBaseWing(aspect_ratio=ar,
                                        taper_ratio=tr,
                                        r22=r22,
                                        camber_angle=16 / 180 * np.pi,
                                        resolution=500)

        configuration.ParamsForMAV_One.change_parameters(sleep_time=0.0001)
        configuration.ParamsForMAV_One.change_parameters(initial_xyz=[0,0,1])
        configuration.ParamsForMAV_One.change_parameters(initial_rpy=[0,0,0])

        # urdf_creator = URDFCreator(gear_ratio=ng,
        #                             aspect_ratio=ar,
        #                             taper_ratio=tr,
        #                             r22=r22,)
        # urdf_name = urdf_creator.write_the_urdf()
        self.temp_urdf = GLOBAL_CONFIGURATION.temporary_urdf_path + f"Ng_{ng}_AR_{ar}_TR_{tr}_R22_{r22}.urdf"


    def free_fly(self,num_mav):

        basemav = BaseMavSequential(
                            urdf_name=self.temp_urdf,
                            mav_params=configuration.ParamsForMAV_One,
                            if_gui=True,
                            if_fixed=False,
        )
        if num_mav%2==0:
            trac_color=[1,0,0]
        else:
            trac_color=[0,0,1]
        self.mav = WrappedMAVpid(basemav,
                        motor_params=self.motor_params,
                        wing_params=self.wing_params,
                        trac_color=trac_color)
        
        obs, wingobs = self.mav.reset()

        self.real_t = 1
        self.durtime = self.real_t*self.mav.CTRL_FREQ
        flag_pid = 0
        cnt = 0
        gradual = 0 
        while cnt < int (self.durtime):

            if (cnt < gradual):
                if flag_pid == 0:
                    r_voltage, l_voltage = self.model.straight_cos()
                    r_voltage = r_voltage * cnt / gradual
                    l_voltage = l_voltage * cnt / gradual
                else:
                    r_voltage, l_voltage = self.model.predict(obs)
                action = [r_voltage, l_voltage]
                obs, wingobs = self.mav.step_no_forceinfo(action=action)

            if (cnt == gradual):
                print("gradual end obs:\n")
                print(obs)

            if (cnt > gradual - 1):
                if flag_pid == 0:
                    r_voltage, l_voltage = self.model.straight_cos()
                else:
                    r_voltage, l_voltage = self.model.predict(obs)
                action = [r_voltage, l_voltage]
                obs, wingobs= self.mav.step_no_forceinfo(action=action)

            if (cnt ==int(0.8*self.durtime+1)):       
                for _ in range(int (3*self.durtime)):
                    self.mav.pause()

            state = np.hstack([obs[0:12],np.zeros(4)])
            otherdata=np.hstack([np.zeros(2), action])
            self.log(cnt, state, otherdata, num_mav)
            cnt = cnt + 1
        self.mav.close()

    def log(self, timestep, state, otherdata, drone=0):
        # log
        # current drone's counter
        current_counter = int(self.counters[drone])
        #
        if(self.timesteps.shape[1]<self.durtime):
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

        #### XYZ ###################################################
        row = 0
        for j in range(self.num_mavs):
            axs[row, col].plot(t,  self.states[j, 0, :], label="mav_" + str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('x (m)')

        row = 1
        for j in range(self.num_mavs):
            axs[row, col].plot(t, self.states[j, 1, :], label="mav_" + str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('y (m)')

        row = 2
        for j in range(self.num_mavs):
            axs[row, col].plot(t,  self.states[j, 2, :], label="mav_" + str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('z (m)')

        #### RPY ###################################################
        row = 3
        for j in range(self.num_mavs):
            axs[row, col].plot(t, (180/np.pi) * self.states[j, 3, :], label="mav_" + str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('r (rad)')
        row = 4
        for j in range(self.num_mavs):
            axs[row, col].plot(t, (180/np.pi) * self.states[j, 4, :], label="mav_" + str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('p (rad)')
        row = 5
        for j in range(self.num_mavs):
            axs[row, col].plot(t, (180/np.pi) * self.states[j, 5, :], label="mav_" + str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('y (rad)')

        #### vx  vy ###################################################
        row = 6
        for j in range(self.num_mavs):
            axs[row, col].plot(t, self.states[j, 6, :], label="mav_" + str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('vx (m/s)')

        row = 7
        for j in range(self.num_mavs):
            axs[row, col].plot(t, self.states[j, 7, :], label="mav_" + str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('vy (m/s)')

        row = 8
        for j in range(self.num_mavs):
            axs[row, col].plot(t, self.otherdata[j, 2, :], label="mav_" + str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('right_stroke_amp (rad)')

        row = 9
        for j in range(self.num_mavs):
            axs[row, col].plot(t, self.otherdata[j, 3, :], label="mav_" + str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('left_stroke_amp (rad)')

        #### Column ################################################
        col = 1

        #### vz ###################################################
        row = 0
        for j in range(self.num_mavs):
            axs[row, col].plot(t, self.states[j, 8, :], label="mav_" + str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('vz (m/s)')

        #### wx wy wz###################################################
        row = 1
        for j in range(self.num_mavs):
            axs[row, col].plot(t, self.states[j, 9, :], label="mav_" + str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('wx (rad/s)')
        row = 2
        for j in range(self.num_mavs):
            axs[row, col].plot(t, self.states[j, 10, :], label="mav_" + str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('wy (rad/s)')
        row = 3
        for j in range(self.num_mavs):
            axs[row, col].plot(t, self.states[j, 11, :], label="mav_" + str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('wz (rad/s)')

        #### U dU U0 sc###################################################
        row = 4
        for j in range(self.num_mavs):
            axs[row, col].plot(t, self.states[j, 12, :], label="mav_" + str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('U (V)')

        row = 5
        for j in range(self.num_mavs):
            axs[row, col].plot(t, self.states[j, 13, :], label="mav_" + str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('dU (V)')

        row = 6
        for j in range(self.num_mavs):
            axs[row, col].plot(t, self.states[j, 14, :], label="mav_" + str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('U0 (V)')

        row = 7
        for j in range(self.num_mavs):
            axs[row, col].plot(t, self.states[j, 15, :], label="mav_" + str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('sc')

        row = 8
        for j in range(self.num_mavs):
            axs[row, col].plot(t, self.otherdata[j, 0, :], label="mav_" + str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('r_u (V)')

        row = 9
        for j in range(self.num_mavs):
            axs[row, col].plot(t, self.otherdata[j, 1, :], label="mav_" + str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('l_u (V)')

        # Drawing options 
        for i in range(8):
            for j in range(2):
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

    def plot_xyzrpy_old(self):
        #### Loop over colors and line styles ######################
        # plt.rc('axes', prop_cycle=(cycler('color', ['r', 'y', 'b']) ))
        plt.rc('axes', prop_cycle=(cycler('color', ['r', 'b', 'r', 'y']) ))
        fig, axs = plt.subplots(3, 2)
        # t = np.arange(0, self.timesteps.shape[1]/GLOBAL_CONFIGURATION.TIMESTEP-1/GLOBAL_CONFIGURATION.TIMESTEP, 1/GLOBAL_CONFIGURATION.TIMESTEP)
        t = np.arange(0, self.timesteps.shape[1])
        t_real = t/1200
        alp_size=20
        tick_size = 20
        #### Column ################################################
        col = 0

        #### XYZ ###################################################
        row = 0
        for j in range(self.num_mavs):
            axs[row, col].plot(t_real,  self.states[j, 0, :]*1000, label="mav_" + str(j))
        axs[row, col].set_xlabel('time(s)',loc='right', fontsize=alp_size)
        axs[row, col].set_ylabel('x (mm)', fontsize=alp_size)
        axs[row, col].set_xticks(np.arange(0, 1.2, 0.2))
        axs[row, col].tick_params(axis='x', labelsize=tick_size )
        axs[row, col].tick_params(axis='y', labelsize=tick_size )

        row = 1
        for j in range(self.num_mavs):
            axs[row, col].plot(t_real, self.states[j, 1, :]*1000 )
        axs[row, col].set_xlabel('time(s)',loc='right', fontsize=alp_size)
        axs[row, col].set_ylabel('y (mm)', fontsize=alp_size)
        axs[row, col].set_xticks(np.arange(0, 1.2, 0.2))
        axs[row, col].tick_params(axis='x', labelsize=tick_size )
        axs[row, col].tick_params(axis='y', labelsize=tick_size )

        row = 2
        for j in range(self.num_mavs):
            axs[row, col].plot(t_real,  self.states[j, 2, :]*1000, label="mav_" + str(j))
        axs[row, col].set_xlabel('time(s)',loc='right', fontsize=alp_size)
        axs[row, col].set_ylabel('z (mm)', fontsize=alp_size)
        axs[row, col].set_xticks(np.arange(0, 1.2, 0.2))
        axs[row, col].tick_params(axis='x', labelsize=tick_size )
        axs[row, col].tick_params(axis='y', labelsize=tick_size )
        axs[row, col].set_yticks([0,500,1000,1500,2000,2500])


        col = 1
        #### RPY ###################################################
        row = 0
        for j in range(self.num_mavs):
            if(j==0):
                axs[row, col].plot(t_real, (180/np.pi)*self.states[j, 3, :], label="$U_a=9V$" )
            else:
                axs[row, col].plot(t_real, (180/np.pi)*self.states[j, 3, :], label="$U_a=12V$" )
        axs[row, col].set_xlabel('time(s)',loc='right', fontsize=alp_size)
        axs[row, col].set_ylabel('roll angle (deg)', fontsize=alp_size)
        axs[row, col].set_xticks(np.arange(0, 1.2, 0.2))
        axs[row, col].tick_params(axis='x', labelsize=tick_size )
        axs[row, col].tick_params(axis='y', labelsize=tick_size )

        row = 1
        for j in range(self.num_mavs):
            axs[row, col].plot(t_real, (180/np.pi)*self.states[j, 4, :], label="mav_" + str(j))
        axs[row, col].set_xlabel('time(s)',loc='right', fontsize=alp_size)
        axs[row, col].set_ylabel('pitch angle (deg)', fontsize=alp_size)
        axs[row, col].set_xticks(np.arange(0, 1.2, 0.2))
        axs[row, col].tick_params(axis='x', labelsize=tick_size )
        axs[row, col].tick_params(axis='y', labelsize=tick_size )

        row = 2
        for j in range(self.num_mavs):
            axs[row, col].plot(t_real, (180/np.pi)*self.states[j, 5, :], label="mav_" + str(j))
        axs[row, col].set_xlabel('time(s)',loc='right', fontsize=alp_size)
        axs[row, col].set_ylabel('yaw angle (deg)', fontsize=alp_size)
        axs[row, col].set_xticks(np.arange(0, 1.2, 0.2))
        axs[row, col].tick_params(axis='x', labelsize=tick_size )
        axs[row, col].tick_params(axis='y', labelsize=tick_size )

        
        # fig.subplots_adjust(left=0.06,
        #                     bottom=0.05,
        #                     right=0.99,
        #                     top=0.98,
        #                     wspace=0.15,
        #                     hspace=0.0
        #                     )
        axs[0,1].legend(fontsize=22, loc='upper right')
        fig.subplots_adjust(left=0.125, right=0.879,top=0.983,bottom=0.09, wspace=0.271,hspace=0.514)
        plt.show()

    def plot_xyzrpy(self):
            #### Loop over colors and line styles ######################
            # plt.rc('axes', prop_cycle=(cycler('color', ['r', 'y', 'b']) ))
            plt.rc('axes', prop_cycle=(cycler('color', ['b']) ))
            fig, axs = plt.subplots(6, 2)
            # t = np.arange(0, self.timesteps.shape[1]/GLOBAL_CONFIGURATION.TIMESTEP-1/GLOBAL_CONFIGURATION.TIMESTEP, 1/GLOBAL_CONFIGURATION.TIMESTEP)
            t = np.arange(0, self.timesteps.shape[1])
            t_real = t/1200
            alp_size=18
            tick_size = 16

            #### XYZ ###################################################
            row = 0
            for j in range(self.num_mavs):
                axs[row,j].plot(t_real,  self.states[j, 0, :]*1000)
                # axs[row,j].set_xlabel('time(s)',loc='right', fontsize=alp_size)
                axs[row,j].set_ylabel('x (mm)', fontsize=alp_size)
                axs[row,j].tick_params(axis='x', labelsize=tick_size )
                axs[row,j].tick_params(axis='y', labelsize=tick_size )
                # axs[row, j].set_xticklabels([])
                if j==0:
                    axs[row,j].set_title('$U_a=12V$',fontsize=alp_size)
                else:
                    axs[row,j].set_title('$U_a=9V$',fontsize=alp_size)
            row = 1
            for j in range(self.num_mavs):
                axs[row,j].plot(t_real, self.states[j, 1, :]*1000 )
                # axs[row,j].set_xlabel('time(s)',loc='right', fontsize=alp_size)
                axs[row,j].set_ylabel('y (mm)', fontsize=alp_size)
                axs[row,j].tick_params(axis='x', labelsize=tick_size )
                axs[row,j].tick_params(axis='y', labelsize=tick_size )
                # axs[row, j].set_xticklabels([])

            row = 2
            for j in range(self.num_mavs):
                axs[row,j].plot(t_real,  self.states[j, 2, :]*1000)
                # axs[row,j].set_xlabel('time(s)',loc='right', fontsize=alp_size)
                axs[row,j].set_ylabel('z (mm)', fontsize=alp_size)
                axs[row,j].tick_params(axis='x', labelsize=tick_size )
                axs[row,j].tick_params(axis='y', labelsize=tick_size )
                # axs[row, j].set_xticklabels([])
            # axs[row].set_yticks([0,0.5,1,1.5,2,2.5])


            #### RPY ###################################################
            row = 3
            for j in range(self.num_mavs):
                axs[row,j].plot(t_real, (180/np.pi)*self.states[j, 1, :] )
                # axs[row,j].set_xlabel('time(s)',loc='right', fontsize=alp_size)
                axs[row,j].set_ylabel('roll (deg)', fontsize=alp_size-2)
                axs[row,j].tick_params(axis='x', labelsize=tick_size )
                axs[row,j].tick_params(axis='y', labelsize=tick_size )
                # axs[row, j].set_xticklabels([])

            row = 4
            for j in range(self.num_mavs):
                axs[row,j].plot(t_real,  (180/np.pi)*self.states[j, 4, :])
                # axs[row,j].set_xlabel('time(s)',loc='right', fontsize=alp_size)
                axs[row,j].set_ylabel('pitch (deg)', fontsize=alp_size-2)
                axs[row,j].tick_params(axis='x', labelsize=tick_size )
                axs[row,j].tick_params(axis='y', labelsize=tick_size )
                # axs[row, j].set_xticklabels([])

            row = 5
            for j in range(self.num_mavs):
                axs[row,j].plot(t_real, (180/np.pi)*self.states[j, 5, :])
                axs[row,j].set_xlabel('time(s)',loc='right', fontsize=alp_size)
                axs[row,j].set_ylabel('yaw (deg)', fontsize=alp_size-2)
                axs[row,j].tick_params(axis='x', labelsize=tick_size )
                axs[row,j].tick_params(axis='y', labelsize=tick_size )

            fig.align_ylabels(axs[:, 0])
            fig.align_ylabels(axs[:, 1])
            # fig.subplots_adjust(left=0.06,
            #                     bottom=0.05,
            #                     right=0.99,
            #                     top=0.98,
            #                     wspace=0.15,
            #                     hspace=0.0
            #                     )
            # axs[0,1].legend(fontsize=22, loc='upper right')
            # fig.subplots_adjust(left=0.125, right=0.879,top=0.983,bottom=0.09, wspace=0.271,hspace=0.348)
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
            handle, = axs.plot(t_real, self.states[j, 0, :]*1000, linewidth=3, label="x", color='tab:red')
            handles.append(handle)
            labels.append("x")

            # Plot y data on second axis
            handle, = axs2.plot(t_real, self.states[j, 1, :]*1000, linewidth=3, label="y", color='tab:green')
            handles.append(handle)
            labels.append("y")

            # Plot z data on third axis
            handle, = axs3.plot(t_real, self.states[j, 2, :]*1000, linewidth=3, label="z", color='tab:blue')
            handles.append(handle)
            labels.append("z")

        # axs.set_xlabel('control step', loc='right', fontsize=alp_size)
        axs.set_xlabel('time(s)', loc='right', fontsize=alp_size)  
        axs.set_ylabel('position (mm)', fontsize=alp_size)
        axs.set_xticks(np.arange(0, self.real_t+1,1))
        axs.set_yticks(np.arange(-100,1200, 100))
        axs2.set_yticks(np.arange(-100,1200, 100))
        axs3.set_yticks(np.arange(-100,1200, 100))
        # axs.set_yticks(np.arange(-100,900, 100))
        # axs2.set_yticks(np.arange(-100,900, 100))
        # axs3.set_yticks(np.arange(-100,900, 100))
        # axs.set_yticks(np.arange(-0.1,1.2, 0.1))
        # axs2.set_yticks(np.arange(-0.1,1.2, 0.1))
        # axs3.set_yticks(np.arange(-0.1, 1.2, 0.1))
        axs.tick_params(axis='x', labelsize=tick_size)
        axs.tick_params(axis='y', labelsize=tick_size)
        axs2.tick_params(axis='y', labelsize=tick_size)
        axs3.tick_params(axis='y', labelsize=tick_size)

        axs.legend(handles, labels, loc='upper right', fontsize=alp_size)
        fig.subplots_adjust(left=0.202, right=0.9,top=0.88,bottom=0.162, wspace=0.2,hspace=0.2)
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
        handle, = axs.plot(t_real, (180/np.pi)*(self.states[j, 4, :]), linewidth=0.5, label="pitch", color='tab:green')
        handles.append(handle)
        labels.append("pitch")

        handle, = axs2.plot(t_real, (180/np.pi)*(self.states[j, 5, :]), linewidth=0.5, label="yaw", color='tab:blue')
        handles.append(handle)
        labels.append("yaw")

        handle, = axs3.plot(t_real, (180/np.pi)*(self.states[j, 3, :]), linewidth=1, label="roll", color='tab:red')
        handles.append(handle)
        labels.append("roll")
        # 将 roll 的 handle 和 label 插入到列表的开头
        handles.insert(0, handle)
        labels.insert(0, "roll")
        handles.pop()
        labels.pop()

        axs.set_xlabel('time(s)', loc='right', fontsize=alp_size)
        axs.set_ylabel('attitude (deg)', fontsize=alp_size)
        axs.set_xticks(np.arange(0, self.real_t+1,1))
        axs.set_yticks(np.arange(-8, 10, 2))
        axs2.set_yticks(np.arange(-8, 10, 2))
        axs3.set_yticks(np.arange(-8, 10, 2))
        # axs.set_yticks(np.arange(-12, 12, 2))
        # axs2.set_yticks(np.arange(-12, 12, 2))
        # axs3.set_yticks(np.arange(-12, 12, 2))
        # axs.set_yticks(np.arange(-15, 55, 5))
        # axs2.set_yticks(np.arange(-15, 55, 5))
        # axs3.set_yticks(np.arange(-15, 55, 5))
        axs.tick_params(axis='x', labelsize=tick_size)
        axs.tick_params(axis='y', labelsize=tick_size)
        axs2.tick_params(axis='y', labelsize=tick_size)
        axs3.tick_params(axis='y', labelsize=tick_size)

        # Use the custom handler for legend
        axs.legend(handles, labels, loc='upper right', fontsize=16, handler_map={Line2D: HandlerLine2DWithWidth()})
        fig.subplots_adjust(left=0.202, right=0.9,top=0.88,bottom=0.162, wspace=0.2,hspace=0.2)
        plt.show()

if __name__ == "__main__":
    freeflytest = free_loop()
    freeflytest.model.hover_voltage=12
    freeflytest.free_fly(0)
    freeflytest.model.hover_voltage=9
    freeflytest.free_fly(1)
    # freeflytest.model.hover_voltage=5
    # freeflytest.free_fly(2)
    freeflytest.plot_xyzrpy()
    # freeflytest.plot_xyz()
    # freeflytest.plot_rpy()
    # freeflytest.plot()

