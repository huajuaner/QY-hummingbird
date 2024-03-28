import numpy as np
from hitsz_qy_hummingbird.configuration.configuration import GLOBAL_CONFIGURATION
from hitsz_qy_hummingbird.base_FWMAV.motor.motor_params import ParamsForBaseMotor
from numpy import pi


class BaseBLDC:

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
        self.v = 0
        self.i = 0
        self.torque_motor = 0
        self.output_power = 0
        self.efficiency3 = 0
        self.data = {
            'current': [],
            'voltage': [],
            'efficiency1': [],
            'efficiency2': [],
            'efficiency3': [],
            'torque_motor': [],
            'output_power': [],
        }

    def step(self,
             voltage,
             stroke_angular_amp,
             stroke_angular_vel,
             stroke_angular_acc,
             if_record=False):
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
        if if_record:
            self.record()
        return torque_output

    def reverse(self,
                torque,
                stroke_angular_amp,
                stroke_angular_vel,
                stroke_angular_acc,
                if_record=False):
        """
        given the output torque,
        calculate the motor voltage and current
        """
        rotor_angular_vel = self.params.gear_ratio * stroke_angular_vel
        rotor_angular_acc = self.params.gear_ratio * stroke_angular_acc

        # this formula has been varified that
        # the spring force is always helping to reduce the torque and
        # the damping force helps to reduce the torque during the decelerating
        output_torque = torque \
                        + self.params.spring_constant * stroke_angular_amp \
                        + self.params.wing_damping_coeff * stroke_angular_vel

        self.output_power = abs(output_torque * stroke_angular_vel)
        self.torque_motor = output_torque / (self.params.gear_ratio * self.params.gear_efficiency)

        self.i = (self.torque_motor
                  + self.params.rotor_damping_coeff * rotor_angular_vel
                  + self.params.rotor_inertia * rotor_angular_acc) \
                 / self.params.torque_constant

        '''
        the current speed constant is in rpm/v
        and the calculated speed constant is in rad/ ( s * v ) 
        '''

        self.v = self.params.terminal_r * self.i \
                 + rotor_angular_vel * 60 / (2 * pi * self.params.speed_constant)
        self.efficiency3 = self.params.torque_constant * rotor_angular_vel / self.v
        if if_record:
            self.record()

    def record(self):
        self.data['current'].append(self.i)
        self.data['voltage'].append(self.v)
        Eff = (self.v * self.i - self.i * self.i * self.params.terminal_r) \
              / (self.v * self.i)
        self.data['efficiency1'].append(Eff)
        self.data['efficiency2'].append(self.output_power / abs(self.i * self.v))
        self.data['efficiency3'].append(self.efficiency3)
        self.data['torque_motor'].append(self.torque_motor)
        self.data['output_power'].append(self.output_power)

    def housekeeping(self):
        for key in self.data.keys():
            self.data[key].clear()
