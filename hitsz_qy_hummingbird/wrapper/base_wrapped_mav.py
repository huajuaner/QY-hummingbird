from hitsz_qy_hummingbird.base_FWMAV.MAV.base_MAV import BaseMAV
from hitsz_qy_hummingbird.base_FWMAV.motor.motor_params import ParamsForBaseMotor
from hitsz_qy_hummingbird.base_FWMAV.wings.wing_params import ParamsForBaseWing
from hitsz_qy_hummingbird.base_FWMAV.wings.base_Wings import BaseWing
from hitsz_qy_hummingbird.base_FWMAV.motor.base_BLDC import BaseBLDC
from hitsz_qy_hummingbird.configuration.configuration import GLOBAL_CONFIGURATION

import numpy as np
import pybullet as p


class WrappedBaseMAV:
    """
    The control step and the simulation step is isolated.
    """

    def __init__(self,
                 mav: BaseMAV,
                 motor_params: ParamsForBaseMotor,
                 wing_params: ParamsForBaseWing):

        self.mav = mav
        self.left_motor = BaseBLDC(motor_params)
        self.right_motor = BaseBLDC(motor_params)
        self.right_wing = BaseWing(wing_params)
        self.left_wing = BaseWing(wing_params)
        self.logger = GLOBAL_CONFIGURATION.logger
        self.data = {}

    def step(self,
             action):
        pass

    def torque_from_voltage(self,
                            action):
        '''
        :param action: right_voltage, left_voltage
        :return: right_torque, left_torque
        '''

        (right_voltage, left_voltage) = action

        (right_stroke_amp, right_stroke_vel, right_stroke_acc, _,
         left_stroke_amp, left_stroke_vel, left_stroke_acc, _) = self.mav.get_state_for_motor_torque()

        right_torque = self.right_motor.step(voltage=right_voltage,
                                             stroke_angular_amp=right_stroke_amp,
                                             stroke_angular_vel=right_stroke_vel,
                                             stroke_angular_acc=right_stroke_acc,
                                             if_record=False)

        left_torque = self.left_motor.step(voltage=left_voltage,
                                           stroke_angular_amp=left_stroke_amp,
                                           stroke_angular_vel=left_stroke_vel,
                                           stroke_angular_acc=left_stroke_acc,
                                           if_record=False)
        return right_torque, left_torque

    def apply_aeroFT(self):
        """
        compute Aerodynamics
        """

        (right_stroke_angular_velocity, right_rotate_angular_velocity,
         right_c_axis, right_r_axis, right_z_axis,
         left_stroke_angular_velocity, left_rotate_angular_velocity,
         left_c_axis, left_r_axis, left_z_axis) = self.mav.get_state_for_wing()

        right_aeroforce, right_pos_bias, right_aerotorque = self.right_wing.calculate_aeroforce_and_torque(
            stroke_angular_velocity=right_stroke_angular_velocity,
            rotate_angular_velocity=right_rotate_angular_velocity,
            r_axis=right_r_axis,
            c_axis=right_c_axis,
            z_axis=right_z_axis
        )

        left_aeroforce, left_pos_bias, left_aerotorque = self.left_wing.calculate_aeroforce_and_torque(
            stroke_angular_velocity=left_stroke_angular_velocity,
            rotate_angular_velocity=left_rotate_angular_velocity,
            r_axis=left_r_axis,
            c_axis=left_c_axis,
            z_axis=left_z_axis
        )

        self.mav.set_link_force_world_frame(
            link_id=self.mav.params.right_wing_link,
            position_bias=right_pos_bias,
            force=right_aeroforce
        )

        self.mav.set_link_torque_world_frame(
            linkid=self.mav.params.right_wing_link,
            torque=right_aerotorque
        )

        self.mav.set_link_force_world_frame(
            link_id=self.mav.params.left_wing_link,
            position_bias=left_pos_bias,
            force=left_aeroforce
        )

        self.mav.set_link_torque_world_frame(
            linkid=self.mav.params.left_wing_link,
            torque=left_aerotorque
        )

    def reset(self):
        self.mav.reset()

    def close(self):
        self.mav.close()

    def if_terminated(self):
        return False

    def if_truncated(self):
        return False

    def housekeeping(self):
        self.mav.housekeeping()
        self.left_motor.housekeeping()
        self.right_motor.housekeeping()
        self.left_wing.housekeeping()
        self.right_wing.housekeeping()
