import numpy as np
import pybullet as p

from hitsz_qy_hummingbird.base_FWMAV.MAV.base_MAV import BaseMAV
from hitsz_qy_hummingbird.base_FWMAV.MAV.MAV_params import ParamsForBaseMAV
from hitsz_qy_hummingbird.base_FWMAV.motor.base_BLDC import BaseBLDC
from hitsz_qy_hummingbird.base_FWMAV.motor.motor_params import ParamsForBaseMotor
from hitsz_qy_hummingbird.base_FWMAV.wings.base_Wings import BaseWing
from hitsz_qy_hummingbird.base_FWMAV.wings.wing_params import ParamsForBaseWing
from hitsz_qy_hummingbird.configuration.configuration import GLOBAL_CONFIGURATION
from hitsz_qy_hummingbird.utils.create_urdf import URDFCreator


class TracMAV():
    """
    This class is specially defined for the test of trajactory pid controller
    in which the controller is outside of the class

    """

    def __init__(self,
                 mav_params: ParamsForBaseMAV,
                 motor_params: ParamsForBaseMotor,
                 wing_params: ParamsForBaseWing):
        # urdf_creator = URDFCreator(gear_ratio=motor_params.gear_ratio,
        #                            r=wing_params.length,
        #                            chord_root=wing_params.chord_root,
        #                            chord_tip=wing_params.chord_tip)
        # urdf_name = urdf_creator.write_the_urdf()

        self.mav = BaseMAV(
                           urdf_name='D://graduate//fwmav//simul2024//240325git//QY-hummingbird//URDFdir//symme0324.urdf',
                           mav_params=mav_params,
                           if_gui=True,
                           if_fixed=False,
                           )
        self.right_motor = BaseBLDC(motor_params)
        self.left_motor = BaseBLDC(motor_params)
        self.right_wing = BaseWing(wing_params, if_log=False)
        self.left_wing = BaseWing(wing_params, if_log=False)

        #frequency
        self.PYB_FREQ = GLOBAL_CONFIGURATION.TIMESTEP
        self.CTRL_FREQ = 1200
        self.PYB_STEPS_PER_CTRL = int (self.PYB_FREQ / self.CTRL_FREQ) 

        #self.logger = GLOBAL_CONFIGURATION.logger
        self.data = {}

    def step(self,
             action):
        """
        One control step, multiple pybullet steps.
        
        param: 
        action: [r_voltage, l_voltage]

        return observation: 
        bs, wingobs

        """
        for _ in range(self.PYB_STEPS_PER_CTRL):
            self.drive_wing(action)
            self.apply_aeroFT()
            self.mav.step()
            GLOBAL_CONFIGURATION.step()

        obs = self.computeObs()
        (right_stroke_amp, right_stroke_vel, right_stroke_acc, _, left_stroke_amp, left_stroke_vel, left_stroke_acc, _) = self.mav.get_state_for_motor_torque()
        wingobs = np.array([right_stroke_amp, left_stroke_amp, right_stroke_vel, left_stroke_vel])
        return  obs, wingobs

    def drive_wing(self, action):
        ''' joint_control of wings' motion.'''
        (r_voltage, l_voltage) = action

        (right_stroke_amp, right_stroke_vel, right_stroke_acc, _,
         left_stroke_amp, left_stroke_vel, left_stroke_acc, _) = self.mav.get_state_for_motor_torque()
        
        r_torque = self.right_motor.step(voltage = r_voltage,
                                        stroke_angular_amp = right_stroke_amp,
                                        stroke_angular_vel = right_stroke_vel,
                                        stroke_angular_acc = right_stroke_acc,
                                        if_record=False )

        l_torque = self.left_motor.step(voltage = l_voltage,
                                        stroke_angular_amp = left_stroke_amp,
                                        stroke_angular_vel = left_stroke_vel,
                                        stroke_angular_acc = left_stroke_acc,
                                        if_record=False )

        self.mav.joint_control(target_right_stroke_amp=None,
                               target_right_stroke_vel=None,
                               right_input_torque= r_torque,
                               target_left_stroke_amp=None,
                               target_left_stroke_vel=None,
                               left_input_torque=l_torque)

    def apply_aeroFT(self):
        '''Compute aerodynamics'''
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

    def computeObs(self):

        self.pos, self.quat = p.getBasePositionAndOrientation(self.mav.body_unique_id, physicsClientId=self.mav.physics_client)
        self.rpy = p.getEulerFromQuaternion(self.quat)
        self.vel, self.ang_v = p.getBaseVelocity(self.mav.body_unique_id, physicsClientId=self.mav.physics_client)

        state = np.hstack((self.pos[:], self.rpy[:],self.vel[:], ))
        #print(state)
        return state.reshape(9,)

    def reset(self):
        self.mav.reset()
        (right_stroke_amp, right_stroke_vel, right_stroke_acc, _,
         left_stroke_amp, left_stroke_vel, left_stroke_acc, _) = self.mav.get_state_for_motor_torque()
        obs = self.computeObs()
        wingobs = np.array([right_stroke_amp, left_stroke_amp, right_stroke_vel, left_stroke_vel])
        return  obs, wingobs

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
