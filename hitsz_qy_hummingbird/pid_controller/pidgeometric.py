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
        l_voltage = -self.generate_control_signal(self.frequency,
                                                        self.voltage_amplitude, 
                                                        self.differential_voltage,  
                                                        self.mean_voltage, 
                                                        -self.split_cycle, 
                                                        GLOBAL_CONFIGURATION.TICKTOCK / GLOBAL_CONFIGURATION.TIMESTEP, 
                                                        0,)
        r_voltage = np.clip(r_voltage, -self.voltage_amplitude_max, self.voltage_amplitude_max)
        l_voltage = np.clip(l_voltage, -self.voltage_amplitude_max, self.voltage_amplitude_max)
        
        return r_voltage, l_voltage

    
    def generate_control_signal(self, f, 
                                Umax, delta, bias, sc, 
                                t, phase_0):
        V = Umax + delta
        V0 = bias
        sigma = 0.5+sc

        T = 1/f
        t_phase = phase_0/360*T
        t = t+t_phase
        period = np.floor(t/T)
        t = t-period*T

        if 0<=t and t<sigma/f:
            u = V*np.cos(2*np.pi*f*(t)/(2*sigma))+V0
        elif sigma/f<=t and t<1/f:
            u = V*np.cos((2*np.pi*f*(t)-2*np.pi)/(2*(1-sigma)))+V0
        else:
            u=0
        return u 

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
        #the current attitude
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        cur_rpy = np.array(p.getEulerFromQuaternion(cur_quat))

        target_quat = (Rotation.from_euler('XYZ', target_euler, degrees=False)).as_quat()
        w,x,y,z = target_quat
        target_rotation = (Rotation.from_quat([w, x, y, z])).as_matrix()

        #rot_matrix_e[2, 1]: This element represents the value at the 3rd row, 2nd column of the target rotation matrix, 
        #corresponding to the roll error in the attitude error.
        #rot_matrix_e[0, 2]: This element represents the value at the 1st row, 3rd column of the target rotation matrix, 
        #corresponding to the pitch error in the attitude error.
        #rot_matrix_e[1, 0]: This element represents the value at the 2nd row, 1st column of the target rotation matrix, 
        #corresponding to the yaw error in the attitude error.
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

        #Using the mixing matrix (self.MIXER_MATRIX) to convert the target torque into PWM signals, np broadcasting thrust.
        pwm = thrust + np.dot(self.MIXER_MATRIX, target_torques)
        pwm = np.clip(pwm, self.MIN_PWM, self.MAX_PWM)

        return self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST