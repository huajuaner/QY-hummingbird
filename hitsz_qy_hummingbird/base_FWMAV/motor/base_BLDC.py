import numpy as np
from hitsz_qy_hummingbird.configuration.configuration import GLOBAL_CONFIGURATION
from hitsz_qy_hummingbird.base_FWMAV.motor.motor_params import ParamsForBaseMotor


class BaseBLDC():
    """
    V                       = R*i
                            + angular_vel / speed_constant

    torque_constant * i     = inertia_of_rotor * angular_acc
                            + rotor_damping_coeff * angular_vel
                            + torque_motor

    torque_output            = torque_motor * gear_ratio * gear_efficiency

    spring_constant         = E   *    d^4   / (67*n*D) *1E2
    """

    def __init__(self,
                 params: ParamsForBaseMotor):
        self.params = params
        self.logger = GLOBAL_CONFIGURATION.logger
        self.data = {}
        self.v = 0
        self.i = 0

    def step(self,
             voltage,
             stroke_angular_amp,
             stroke_angular_vel,
             stroke_angular_acc):
        """
        Given the Voltage
        return the output torque which has already considered the spring effect
        """
        rotor_angular_vel = stroke_angular_vel * self.params.gear_ratio
        rotor_angular_acc = stroke_angular_acc * self.params.gear_ratio
        self.v = voltage
        self.i = (voltage - rotor_angular_vel / self.params.speed_constant) / self.params.terminal_r

        torque_motor = (self.i * self.params.torque_constant
                        - self.params.rotor_inertia * rotor_angular_acc
                        - self.params.rotor_damping_coeff * rotor_angular_vel)

        torque_output = (torque_motor * self.params.gear_ratio * self.params.gear_efficiency
                         - self.params.spring_constant * stroke_angular_amp
                         - self.params.wing_damping_coeff * stroke_angular_vel)

        return torque_output

    def reverse(self,
                torque,
                stroke_angular_amp,
                stroke_angular_vel,
                stroke_angular_acc):
        """
        given the output torque,
        calculate the motor voltage and current
        """
        rotor_angular_vel = self.params.gear_ratio * stroke_angular_vel
        rotor_angular_acc = self.params.gear_ratio * stroke_angular_acc

        torque_motor = (torque
                        + self.params.spring_constant * stroke_angular_amp
                        + self.params.wing_damping_coeff * stroke_angular_vel) \
                       / (self.params.gear_ratio * self.params.gear_efficiency)

        self.i = (torque_motor
                  + self.params.rotor_damping_coeff * rotor_angular_vel
                  + self.params.rotor_inertia * rotor_angular_acc) \
                 / self.params.torque_constant

        self.v = self.params.terminal_r * self.i \
                 + rotor_angular_vel / self.params.speed_constant

    def housekeeping(self):
        self.data.clear()

