import numpy as np

from hitsz_qy_hummingbird.pid_controller.a_pidobject import PidObj
from hitsz_qy_hummingbird.configuration.configuration import GLOBAL_CONFIGURATION


class PIDLinear():

    def __init__(self):
        # target
        self.pos_target_x = 0
        self.pos_target_y = 0
        self.pos_target_z = 0
        self.ang_ef_target_yaw = 0

        # current measured pos, vel
        self.pos_current_x = 0
        self.pos_current_y = 0
        self.pos_current_z = 0

        self.vel_current_x = 0
        self.vel_current_y = 0
        self.vel_current_z = 0

        # current measured rpy
        self.ang_cur_roll = 0
        self.ang_cur_pitch = 0
        self.ang_cur_yaw = 0

        # control limit
        self.differential_voltage_max = 1.5
        self.mean_voltage_max = 1.5
        self.split_cycle_max = 0.05
        self.voltage_amplitude_max = 20


        self.hover_voltage = 15
        self.frequency = 28

        self.d_voltage_amplitude = 0
        self.voltage_amplitude = 0
        self.differential_voltage = 0
        self.mean_voltage = 0
        self.split_cycle = 0

        self.acc_target_xy_max = 10
        # xy postion to velocity
        self.p_pos_xy_Kp_ = 0
        # z postion to velocity
        self.p_pos_z_Kp = 0

    def predict(self, observation):

        self.pos_current_x = observation[0]
        self.pos_current_y = observation[1]
        self.pos_current_z = observation[2]

        self.ang_cur_roll = observation[3]
        self.ang_cur_pitch = observation[4]
        self.ang_cur_yaw = observation[5]

        self.vel_current_x = observation[6]
        self.vel_current_y = observation[7]
        self.vel_current_z = observation[8]

        self.controller_run()

        r_voltage = self.generate_control_signal(self.frequency,
                                                 self.voltage_amplitude,
                                                 self.differential_voltage,
                                                 self.mean_voltage,
                                                 -self.split_cycle,
                                                 GLOBAL_CONFIGURATION.TICKTOCK / GLOBAL_CONFIGURATION.TIMESTEP,
                                                 0, )
        l_voltage = -self.generate_control_signal(self.frequency,
                                                  self.voltage_amplitude,
                                                  -self.differential_voltage,
                                                  self.mean_voltage,
                                                  self.split_cycle,
                                                  GLOBAL_CONFIGURATION.TICKTOCK / GLOBAL_CONFIGURATION.TIMESTEP,
                                                  0, )
        r_voltage = np.clip(r_voltage, -self.voltage_amplitude_max, self.voltage_amplitude_max)
        l_voltage = np.clip(l_voltage, -self.voltage_amplitude_max, self.voltage_amplitude_max)

        return r_voltage, l_voltage

    def straight_cos(self):
        self.voltage_amplitude = self.hover_voltage + self.d_voltage_amplitude
        r_voltage = self.generate_control_signal(self.frequency,
                                                 self.voltage_amplitude,
                                                 self.differential_voltage,
                                                 self.mean_voltage,
                                                 self.split_cycle,
                                                 GLOBAL_CONFIGURATION.TIME,
                                                 0, )
        l_voltage = - self.generate_control_signal(self.frequency,
                                                   self.voltage_amplitude,
                                                   -self.differential_voltage,
                                                   self.mean_voltage,
                                                   -self.split_cycle,
                                                   GLOBAL_CONFIGURATION.TIME,
                                                   0, )
        return r_voltage, l_voltage

    def generate_control_signal(self, f,
                                Umax, delta, bias, sc,
                                t, phase_0):
        V = Umax + delta
        V0 = bias
        sigma = 0.5 + sc

        T = 1 / f
        t_phase = phase_0 / 360 * T
        t = t + t_phase
        period = np.floor(t / T)
        t = t - period * T

        if 0 <= t and t < sigma / f:
            u = V * np.cos(2 * np.pi * f * (t) / (2 * sigma)) + V0
        elif sigma / f <= t and t < 1 / f:
            u = V * np.cos((2 * np.pi * f * (t) - 2 * np.pi) / (2 * (1 - sigma))) + V0
        else:
            u = 0
        return u

    def controller_run(self):
        self.xy_control()
        self.attitude_control()
        self.z_control()

    def xy_control(self):

        self.p_pos_xy_Kp_ = 5
        pos_error_x = self.pos_target_x - self.pos_current_x
        pos_error_y = self.pos_target_y - self.pos_current_y
        vel_target_x = self.p_pos_xy_Kp_ * pos_error_x
        vel_target_y = self.p_pos_xy_Kp_ * pos_error_y

        acc_pid_x = PidObj(
            integ_max=10,
            Kp=0.2,
            Ki=0.05,
            Kd=0.002,  # 0.01,
        )
        vel_error_x = vel_target_x - self.vel_current_x
        acc_target_x = acc_pid_x.pid_update(vel_error_x)
        acc_target_x = np.clip(acc_target_x, -self.acc_target_xy_max, self.acc_target_xy_max)

        acc_pid_y = PidObj(
            integ_max=10,
            Kp=0.2,
            Ki=0.05,
            Kd=0.002,  # 0.002,
        )
        vel_error_y = vel_target_y - self.vel_current_y

        acc_target_y = acc_pid_y.pid_update(vel_error_y)
        acc_target_y = np.clip(acc_target_y, -self.acc_target_xy_max, self.acc_target_xy_max)

        # These two lines of code perform coordinate system rotation, converting the target accelerations acc_target_x and acc_target_y
        # from the global coordinate system (usually Earth coordinates) to the aircraft body coordinate system.
        acc_fwd = acc_target_x * np.cos(self.ang_cur_yaw) + acc_target_y * np.sin(self.ang_cur_yaw)
        acc_lft = -acc_target_x * np.sin(self.ang_cur_yaw) + acc_target_y * np.cos(self.ang_cur_yaw)

        # These two lines of code use the arctan function to map the rotated accelerations to pitch and roll angles.
        # In planar models, aircraft motion is typically constrained to the horizontal plane, 
        # neglecting its motion in the vertical direction.
        # This means that changes in the aircraft's attitude, climb, and descent in the vertical direction are ignored.
        self.ang_ef_target_pitch = np.arctan(acc_fwd / 9.81)
        self.ang_ef_target_roll = -np.arctan(acc_lft * np.cos(self.ang_ef_target_pitch) / 9.81)

    def z_control(self):

        self.p_pos_z_Kp = 5
        pos_error_z = self.pos_target_z - self.pos_current_z
        vel_target_z = pos_error_z * self.p_pos_z_Kp

        voltage_pid = PidObj(
            integ_max=10,
            Kp=10,
            Ki=5,
            Kd=0.05,  # 0.04,
        )
        vel_error_z = vel_target_z - self.vel_current_z
        voltage_out = voltage_pid.pid_update(vel_error_z)

        voltage_out = voltage_out / (np.cos(self.ang_cur_pitch) * np.cos(self.ang_cur_roll))
        self.voltage_amplitude = np.clip(voltage_out, 0, self.voltage_amplitude_max - self.hover_voltage)
        self.voltage_amplitude += self.hover_voltage

    def attitude_control(self):

        ang_ef_target_roll = np.clip(self.ang_ef_target_roll, -30.0 / 180.0 * np.pi, 30.0 / 180.0 * np.pi)
        ang_ef_target_pitch = np.clip(self.ang_ef_target_pitch, -30.0 / 180.0 * np.pi, 30.0 / 180.0 * np.pi)

        ang_ef_error_roll = self.wrap_180(ang_ef_target_roll - self.ang_cur_roll)
        ang_ef_error_pitch = self.wrap_180(ang_ef_target_pitch - self.ang_cur_pitch)
        ang_ef_error_yaw = self.wrap_180(self.ang_ef_target_yaw - self.ang_cur_yaw)
        ang_ef_error_yaw = np.clip(ang_ef_error_yaw, -10 / 180 * np.pi, 10 / 180 * np.pi)

        # convert to body frame
        ang_bf_error = self.frame_ef_to_bf(ang_ef_error_roll, ang_ef_error_pitch, ang_ef_error_yaw)
        ang_bf_error_x = ang_bf_error[0]
        ang_bf_error_y = ang_bf_error[1]
        ang_bf_error_z = ang_bf_error[2]

        pid_roll = PidObj(
            integ_max=2,
            Kp=5,  # 3,
            Ki=1.5,
            Kd=0.3,  # 0.3,
        )
        self.differential_voltage = pid_roll.pid_update(ang_bf_error_x)
        self.differential_voltage = np.clip(self.differential_voltage, -self.differential_voltage_max,
                                            self.differential_voltage_max)

        pid_pitch = PidObj(
            integ_max=2,
            Kp=5,  # 5,
            Ki=2,
            Kd=0.5,  # 0.4,
        )
        self.mean_voltage = pid_pitch.pid_update(ang_bf_error_y)
        self.mean_voltage = np.clip(self.mean_voltage, -self.mean_voltage_max, self.mean_voltage_max)

        pid_yaw = PidObj(
            integ_max=1,
            Kp=0.2,
            Ki=0.3,
            Kd=0.003,
        )
        self.split_cycle = pid_yaw.pid_update(ang_bf_error_z)
        self.split_cycle = np.clip(self.split_cycle, -self.split_cycle_max, self.split_cycle_max)

    def wrap_180(self, angle):
        # Restrict the given angle to the range from -π to π.
        if (angle > 3 * np.pi or angle < -3 * np.pi):
            angle = np.fmod(angle, 2 * np.pi)
        if (angle > np.pi):
            angle = angle - 2 * np.pi
        if (angle < - np.pi):
            angle = angle + 2 * np.pi
        return angle

    def frame_ef_to_bf(self, ef_x, ef_y, ef_z):
        """
        It takes the three-dimensional vectors ef_x, ef_y, and ef_z from EF as inputs,
        then performs coordinate transformation based on the attitude information of the FWMAV
        (pitch angle, roll angle, yaw angle), obtaining the corresponding vector in BF.
        """
        bf = np.zeros([3], dtype=np.float64)
        self.cos_r = np.cos(self.ang_cur_roll)
        self.sin_r = np.sin(self.ang_cur_roll)
        self.cos_p = np.cos(self.ang_cur_pitch)
        self.sin_p = np.sin(self.ang_cur_pitch)

        bf[0] = ef_x - self.sin_p * ef_z
        bf[1] = self.cos_r * ef_y + self.sin_r * self.cos_p * ef_z
        bf[2] = -self.sin_r * ef_y + self.cos_p * self.cos_r * ef_z
        return bf
