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
                           if_fixed=False)
        self.right_motor = BaseBLDC(motor_params)
        self.left_motor = BaseBLDC(motor_params)
        self.right_wing = BaseWing(wing_params)
        self.left_wing = BaseWing(wing_params)

        self.logger = GLOBAL_CONFIGURATION.logger
        self.data = {}
        self.observation = None

    def step(self,
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

        (right_stroke_amp,
         right_stroke_vel,
         right_input_torque,
         left_stroke_amp,
         left_stroke_vel,
         left_input_torque) = action

        self.mav.joint_control(target_right_stroke_amp=right_stroke_amp,
                               target_right_stroke_vel=right_stroke_vel,
                               right_input_torque=right_input_torque,
                               target_left_stroke_amp=left_stroke_amp,
                               target_left_stroke_vel=left_stroke_vel,
                               left_input_torque=left_input_torque)

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
            linkid= self.mav.params.left_wing_link,
            torque=left_aerotorque
        )

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

        return  [right_stroke_amp, right_stroke_vel,
                right_stroke_acc, right_torque,
                right_motor_current, right_motor_voltage,
                left_stroke_amp, left_stroke_vel,
                left_stroke_acc, left_torque,
                left_motor_current, left_motor_voltage] + list(force_info)

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
