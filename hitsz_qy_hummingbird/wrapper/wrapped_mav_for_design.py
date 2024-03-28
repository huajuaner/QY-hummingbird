from hitsz_qy_hummingbird.base_FWMAV.MAV.base_MAV_sequential import BaseMavSequential
from hitsz_qy_hummingbird.base_FWMAV.MAV.MAV_params import ParamsForBaseMAV
from hitsz_qy_hummingbird.base_FWMAV.motor.base_BLDC import BaseBLDC
from hitsz_qy_hummingbird.base_FWMAV.motor.motor_params import ParamsForBaseMotor
from hitsz_qy_hummingbird.base_FWMAV.wings.base_Wings import BaseWing
from hitsz_qy_hummingbird.base_FWMAV.wings.wing_params import ParamsForBaseWing
from hitsz_qy_hummingbird.configuration.configuration import GLOBAL_CONFIGURATION
from hitsz_qy_hummingbird.wrapper.base_wrapped_mav import WrappedBaseMAV
from hitsz_qy_hummingbird.utils.create_urdf import URDFCreator


class WrappedMAVDesign(WrappedBaseMAV):
    """
    This class is specially defined for the test of wing beat controller
    in which the controller is outside the class
    """

    # Init Function is directly adopted from WrappedBaseMAV

    def joint_control_amp(self,
                          action):
        """
        :param action: [
            right_stroke_amp
            right_stroke_vel
            right_input_torque
            left_stroke_amp
            left_stroke_vel
            left_input_torque
            ]
        """
        (target_right_stroke_amp,
         target_right_stroke_vel,
         target_right_input_torque,
         target_left_stroke_amp,
         target_left_stroke_vel,
         target_left_input_torque) = action

        self.mav.joint_control(target_right_stroke_amp=target_right_stroke_amp,
                               target_right_stroke_vel=target_right_stroke_vel,
                               right_input_torque=target_right_input_torque,
                               target_left_stroke_amp=target_left_stroke_amp,
                               target_left_stroke_vel=target_left_stroke_vel,
                               left_input_torque=target_left_input_torque)

    def joint_control_voltage(self,
                              action):
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
        return right_torque,left_torque

    def step_after_joint_control(self):
        """
        return observation: [
            right_stroke_amp    right_stroke_vel    right_stroke_acc
            right_input_torque  right_motor_current right_motor_voltage
            left_stroke_amp     left_stroke_vel     left_stroke_acc
            left_input_torque   left_motor_current  left_motor_voltage
            BodyForce1          BodyForce2          BodyForce3
            BodyTorque1         BodyTorque2         BodyTorque3
            ]
        """
        WrappedBaseMAV.apply_aeroFT(self)
        self.mav.step()
        (right_stroke_amp, right_stroke_vel, right_stroke_acc, right_torque,
         left_stroke_amp, left_stroke_vel, left_stroke_acc, left_torque) = self.mav.get_state_for_motor_torque()
        self.right_motor.reverse(torque=right_torque,
                                 stroke_angular_amp=right_stroke_amp,
                                 stroke_angular_vel=right_stroke_vel,
                                 stroke_angular_acc=right_stroke_acc)

        right_motor_current = self.right_motor.i
        right_motor_voltage = self.right_motor.v

        self.left_motor.reverse(torque=left_torque,
                                stroke_angular_amp=left_stroke_amp,
                                stroke_angular_vel=left_stroke_vel,
                                stroke_angular_acc=left_stroke_acc)

        left_motor_current = self.left_motor.i
        left_motor_voltage = self.left_motor.v

        force_info = self.mav.get_constraint_state()

        return [right_stroke_amp, right_stroke_vel,
                right_stroke_acc, right_torque,
                right_motor_current, right_motor_voltage,
                left_stroke_amp, left_stroke_vel,
                left_stroke_acc, left_torque,
                left_motor_current, left_motor_voltage] + list(force_info)
