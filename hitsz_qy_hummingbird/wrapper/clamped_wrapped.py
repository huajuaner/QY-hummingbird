from hitsz_qy_hummingbird.base_FWMAV.MAV.base_MAV import BaseMAV
from hitsz_qy_hummingbird.base_FWMAV.MAV.MAV_params import ParamsForBaseMAV
from hitsz_qy_hummingbird.base_FWMAV.motor.base_BLDC import BaseBLDC
from hitsz_qy_hummingbird.base_FWMAV.motor.motor_params import ParamsForBaseMotor
from hitsz_qy_hummingbird.base_FWMAV.wings.base_Wings import BaseWing
from hitsz_qy_hummingbird.base_FWMAV.wings.wing_params import ParamsForBaseWing
from hitsz_qy_hummingbird.configuration.configuration import GLOBAL_CONFIGURATION
from hitsz_qy_hummingbird.utils.create_urdf import URDFCreator


class ClampedMAV():
    """
    This class is specially defined for the test of wing beat controller
    in which the controller is outside of the class
    """

    def __init__(self,
                 mav_params: ParamsForBaseMAV,
                 motor_params: ParamsForBaseMotor,
                 wing_params: ParamsForBaseWing):

        urdf_creator = URDFCreator(gear_ratio=motor_params.gear_ratio,
                                   r=wing_params.length,
                                   chord_root=wing_params.chord_root,
                                   chord_tip=wing_params.chord_tip)
        urdf_name = urdf_creator.write_the_urdf()

        self.mav = BaseMAV(urdf_name=urdf_name,
                           mav_params=mav_params,
                           if_gui=True,
                           if_fixed=True)
        self.right_motor = BaseBLDC(motor_params)
        self.left_motor = BaseBLDC(motor_params)
        self.right_wing = BaseWing(wing_params)
        self.left_wing = BaseWing(wing_params)

        self.data={}
        self.observation = None

    def step(self,
             action):
        """
        :param action: [
            right_stroke_amp    right_stroke_vel    right_input_torque
            left_stroke_amp     left_stroke_vel     left_stroke_torque
            ]
        return observation: [
            right_stroke_amp    right_stroke_vel    right_stroke_acc
            right_input_torque  right_motor_current right_motor_voltage
            left_stroke_amp     left_stroke_vel     left_stroke_acc
            left_input_torque   left_motor_current  left_motor_voltage
            BodyForce1          BodyForce2          BodyForce3
            BodyTorque1         BodyTorque2         BodyTorque3
            ]
            terminated: bool
            truncated: bool
        """
        (right_stroke_amp, right_stroke_vel, right_input_torque,
         left_stroke_amp, left_stroke_vel, left_stroke_torque) = action


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