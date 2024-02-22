import os
import numpy as np
import xml.etree.ElementTree as etxml
import pkg_resources
import math
import pybullet as p
from scipy.spatial.transform import Rotation
from hitsz_qy_hummingbird.configuration.configuration import GLOBAL_CONFIGURATION

class PIDgeo():

    def __init__(self):
        # target
        self.pos_target_x = 0
        self.pos_target_y = 0
        self.pos_target_z = 0
        self.ang_ef_target_yaw = 0 

        #current measured pos, vel 
        self.pos_current_x = 0
        self.pos_current_y = 0
        self.pos_current_z = 0

        self.vel_current_x = 0
        self.vel_current_y = 0
        self.vel_current_z = 0

        #current measured rpy
        self.ang_cur_roll = 0
        self.ang_cur_pitch = 0
        self.ang_cur_yaw = 0

        #control limit
        self.differential_voltage_max = 0.5
        self.mean_voltage_max = 3.5
        self.split_cycle_max = 0.05
        self.hover_voltage = 12
        self.voltage_amplitude_max = 18

        self.frequency = 34
        self.voltage_amplitude = 12
        self.differential_voltage = 0
        self.mean_voltage = 0
        self.split_cycle = 0

        self.acc_target_xy_max = 10
        # xy postion to velocity
        self.p_pos_xy_Kp_ = 7	
        # z postion to velocity
        self.p_pos_z_Kp = 5

    ################################################################################

    def reset(self):
        """Reset the control classes.

        A general use counter is set to zero.

        """
        self.control_counter = 0

    ################################################################################

    def predict(self, observation):

        self.pos_current_x = observation[0]
        self.pos_current_y = observation[1]
        self.pos_current_z = observation[2]

        self.ang_cur_roll =  observation[3]
        self.ang_cur_pitch = observation[4]
        self.ang_cur_yaw =   observation[5]
        
        self.vel_current_x = observation[6]
        self.vel_current_y = observation[7]
        self.vel_current_z = observation[8]

        self.controller_run()

        r_voltage = self.generate_control_signal(self.frequency, 
                                                        self.voltage_amplitude, 
                                                        -self.differential_voltage,  
                                                        self.mean_voltage, 
                                                        self.split_cycle, 
                                                        GLOBAL_CONFIGURATION.TICKTOCK / GLOBAL_CONFIGURATION.TIMESTEP , 
                                                        0,)
        l_voltage = self.generate_control_signal(self.frequency,
                                                        self.voltage_amplitude, 
                                                        self.differential_voltage,  
                                                        self.mean_voltage, 
                                                        -self.split_cycle, 
                                                        GLOBAL_CONFIGURATION.TICKTOCK / GLOBAL_CONFIGURATION.TIMESTEP, 
                                                        0,)
        r_voltage = np.clip(r_voltage, -18, 18)
        l_voltage = np.clip(l_voltage, -18, 18)
        
        return r_voltage, -l_voltage

    ################################################################################

    def computeControl(self,
                       control_timestep,
                       cur_pos,
                       cur_quat,
                       cur_vel,
                       cur_ang_vel,
                       target_pos,
                       target_rpy=np.zeros(3),
                       target_vel=np.zeros(3),
                       target_rpy_rates=np.zeros(3)
                       ):
        self.control_counter += 1
        thrust, computed_target_rpy, pos_e = self.PIDPositionControl(control_timestep,
                                                                         cur_pos,
                                                                         cur_quat,
                                                                         cur_vel,
                                                                         target_pos,
                                                                         target_rpy,
                                                                         target_vel
                                                                         )
        rpm = self.PIDAttitudeControl(control_timestep,
                                          thrust,
                                          cur_quat,
                                          computed_target_rpy,
                                          target_rpy_rates
                                          )
        cur_rpy = p.getEulerFromQuaternion(cur_quat)
        return rpm, pos_e, computed_target_rpy[2] - cur_rpy[2]

################################################################################
    def PIDPositionControl(self,
                               control_timestep,
                               cur_pos,
                               cur_quat,
                               cur_vel,
                               target_pos,
                               target_rpy,
                               target_vel
                               ):
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        pos_e = target_pos - cur_pos
        vel_e = target_vel - cur_vel

        self.integral_pos_e = self.integral_pos_e + pos_e*control_timestep
        self.integral_pos_e = np.clip(self.integral_pos_e, -2., 2.)
        self.integral_pos_e[2] = np.clip(self.integral_pos_e[2], -0.15, .15)

        #### PID target thrust #####################################
        target_thrust = np.multiply(self.P_COEFF_FOR, pos_e) \
                        + np.multiply(self.I_COEFF_FOR, self.integral_pos_e) \
                        + np.multiply(self.D_COEFF_FOR, vel_e) + np.array([0, 0, self.GRAVITY])
        
        scalar_thrust = max(0., np.dot(target_thrust, cur_rotation[:,2]))
        thrust = (math.sqrt(scalar_thrust / (4*self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE

        target_z_ax = target_thrust / np.linalg.norm(target_thrust)
        target_x_c = np.array([math.cos(target_rpy[2]), math.sin(target_rpy[2]), 0])
        target_y_ax = np.cross(target_z_ax, target_x_c) / np.linalg.norm(np.cross(target_z_ax, target_x_c))
        target_x_ax = np.cross(target_y_ax, target_z_ax)
        target_rotation = (np.vstack([target_x_ax, target_y_ax, target_z_ax])).transpose()
        
        #### Target rotation #######################################
        target_euler = (Rotation.from_matrix(target_rotation)).as_euler('XYZ', degrees=False)
        if np.any(np.abs(target_euler) > math.pi):
            print("\n[ERROR] ctrl it", self.control_counter, "in Control._dslPIDPositionControl(), values outside range [-pi,pi]")
        return thrust, target_euler, pos_e
    
    ################################################################################

    def PIDAttitudeControl(self,
                               control_timestep,
                               thrust,
                               cur_quat,
                               target_euler,
                               target_rpy_rates
                               ):
        #当前姿态表示
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        cur_rpy = np.array(p.getEulerFromQuaternion(cur_quat))

        target_quat = (Rotation.from_euler('XYZ', target_euler, degrees=False)).as_quat()
        w,x,y,z = target_quat
        target_rotation = (Rotation.from_quat([w, x, y, z])).as_matrix()

        #rot_matrix_e[2, 1]: 这个元素表示目标旋转矩阵的第 3 行、第 2 列的值，对应于姿态误差中的横滚（roll）误差。
        #rot_matrix_e[0, 2]: 这个元素表示目标旋转矩阵的第 1 行、第 3 列的值，对应于姿态误差中的俯仰（pitch）误差。
        #rot_matrix_e[1, 0]: 这个元素表示目标旋转矩阵的第 2 行、第 1 列的值，对应于姿态误差中的偏航（yaw）误差。
        rot_matrix_e = np.dot((target_rotation.transpose()),cur_rotation) - np.dot(cur_rotation.transpose(),target_rotation)
        rot_e = np.array([rot_matrix_e[2, 1], rot_matrix_e[0, 2], rot_matrix_e[1, 0]]) 

        rpy_rates_e = target_rpy_rates - (cur_rpy - self.last_rpy)/control_timestep
        self.last_rpy = cur_rpy

        self.integral_rpy_e = self.integral_rpy_e - rot_e*control_timestep
        self.integral_rpy_e = np.clip(self.integral_rpy_e, -1500., 1500.)
        self.integral_rpy_e[0:2] = np.clip(self.integral_rpy_e[0:2], -1., 1.)

        #### PID target torques ####################################
        target_torques = - np.multiply(self.P_COEFF_TOR, rot_e) \
                         + np.multiply(self.D_COEFF_TOR, rpy_rates_e) \
                         + np.multiply(self.I_COEFF_TOR, self.integral_rpy_e)
        target_torques = np.clip(target_torques, -3200, 3200)

        #使用混合矩阵（self.MIXER_MATRIX）将目标力矩转换为PWM信号,np广播thrust
        pwm = thrust + np.dot(self.MIXER_MATRIX, target_torques)
        pwm = np.clip(pwm, self.MIN_PWM, self.MAX_PWM)

        #将PWM信号转换为电机转速(RPM)
        return self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST