"""
This MAV is driven in voltage
the mav is
"""

import numpy as np
import pybullet as p

from hitsz_qy_hummingbird.base_FWMAV.MAV.base_MAV import BaseMAV
from hitsz_qy_hummingbird.wrapper.base_wrapped_mav import WrappedBaseMAV
# from hitsz_qy_hummingbird.base_FWMAV.MAV.base_MAV_sequential import BaseMavSequential
from hitsz_qy_hummingbird.base_FWMAV.MAV.MAV_params import ParamsForBaseMAV
from hitsz_qy_hummingbird.base_FWMAV.motor.base_BLDC import BaseBLDC
from hitsz_qy_hummingbird.base_FWMAV.motor.motor_params import ParamsForBaseMotor
from hitsz_qy_hummingbird.base_FWMAV.wings.base_Wings import BaseWing
from hitsz_qy_hummingbird.base_FWMAV.wings.wing_params import ParamsForBaseWing
from hitsz_qy_hummingbird.configuration.configuration import GLOBAL_CONFIGURATION
from hitsz_qy_hummingbird.utils.create_urdf import URDFCreator


class WrappedMAVRl(WrappedBaseMAV):
    """
    This class is specially defined for the test of trajectory pid controller
    in which the controller is outside the class
    """

    def __init__(self,
                 mav: BaseMAV,
                 motor_params: ParamsForBaseMotor,
                 wing_params: ParamsForBaseWing,
                 control_frequency=1200):
        super().__init__(mav,
                         motor_params,
                         wing_params)

        # frequency
        self.PYB_FREQ = GLOBAL_CONFIGURATION.TIMESTEP
        self.CTRL_FREQ = control_frequency
        self.PYB_STEPS_PER_CTRL = int(self.PYB_FREQ / self.CTRL_FREQ)

    def step(self,
             action):
        """
        One control step, multiple pybullet steps.
        
        param: 
        action: [r_voltage, l_voltage]

        return observation: 
        obs, wingobs

        """
        for _ in range(self.PYB_STEPS_PER_CTRL):
            self.drive_wing(action)
            WrappedBaseMAV.apply_aeroFT(self)
            self.mav.step()
            GLOBAL_CONFIGURATION.step()

        obs, wingobs = self.computeObsandWingObs()
        return obs, wingobs

    def drive_wing(self, action):
        """
        :param action: right_voltage, left_voltage
        """
        right_torque, left_torque = WrappedBaseMAV.torque_from_voltage(self,
                                                                       action=action)
        self.mav.joint_control(target_right_stroke_amp=None,
                               target_right_stroke_vel=None,
                               right_input_torque=right_torque,
                               target_left_stroke_amp=None,
                               target_left_stroke_vel=None,
                               left_input_torque=left_torque)

    def computeObsandWingObs(self):
        pos, quat = p.getBasePositionAndOrientation(self.mav.body_unique_id,
                                                    physicsClientId=self.mav.physics_client)
        rpy = p.getEulerFromQuaternion(quat)
        vel, ang_v = p.getBaseVelocity(self.mav.body_unique_id,
                                       physicsClientId=self.mav.physics_client)
        # TODO: can this code work for both mav?


        state = np.hstack((pos[:],
                           rpy[:],
                           vel[:],))
        (right_stroke_amp, right_stroke_vel, right_stroke_acc, _,
         left_stroke_amp, left_stroke_vel, left_stroke_acc, _) = self.mav.get_state_for_motor_torque()
        wingobs = np.array([right_stroke_amp, left_stroke_amp, right_stroke_vel, left_stroke_vel])
        return state.reshape(9, ), wingobs

    def reset(self):
        self.mav.reset()
        obs, wingobs = self.computeObsandWingObs()
        return obs, wingobs
