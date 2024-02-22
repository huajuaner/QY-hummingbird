import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces

from hitsz_qy_hummingbird.configuration import configuration
from hitsz_qy_hummingbird.base_FWMAV.MAV.baserl_MAV import BaseRlMAV
from hitsz_qy_hummingbird.base_FWMAV.MAV.MAV_params import ParamsForBaseMAV
from hitsz_qy_hummingbird.base_FWMAV.motor.base_BLDC import BaseBLDC
from hitsz_qy_hummingbird.base_FWMAV.motor.motor_params import ParamsForBaseMotor
from hitsz_qy_hummingbird.base_FWMAV.wings.base_Wings import BaseWing
from hitsz_qy_hummingbird.base_FWMAV.wings.wing_params import ParamsForBaseWing
from hitsz_qy_hummingbird.configuration.configuration import GLOBAL_CONFIGURATION
from hitsz_qy_hummingbird.utils.create_urdf import URDFCreator


class RLMAV(gym.Env):
    """
    This class is specially defined for the test of RL

    """

    def __init__(self,
                 mav_params = configuration.ParamsForMAV_rl,
                 motor_params = configuration.ParamsForMaxonSpeed6M_rl,
                 wing_params = configuration.ParamsForWing_rl,
                 gui = True
                 ):

        if gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        self.mav_params = mav_params
        self.motor_params = motor_params
        self.wing_params = wing_params

        self._housekeeping(self.physics_client)

        #### Create action and observation spaces ##################
        self.max_angle_of_stroke = np.pi * 0.4
        self.action_space = self._actionSpace()
        self.observation_space = self._observationSpace()

        #frequency
        self.PYB_FREQ = GLOBAL_CONFIGURATION.TIMESTEP
        self.CTRL_FREQ = 24000
        self.EPISODE_LEN_SEC = 0.5

    def step(self,
             action):
        """
        
        """

        (right_stroke_amp,
         left_stroke_amp) = self._preAction(action)

        self.mav.joint_control(target_right_stroke_amp=right_stroke_amp,
                               target_left_stroke_amp=left_stroke_amp)

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
        self.step_counter =  GLOBAL_CONFIGURATION.TICKTOCK

        obs = self._computeObs()
        reward = self._computeReward()
        terminated = self._computeTerminated()
        truncated = self._computeTruncated()
        info = self._computeInfo()

        return obs, reward, terminated, truncated, info

    def geo(self,
            action):
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
        self.mav.step()

    def reset(self,
              seed : int = None,):
        """Resets the environment.

        Parameters
        ----------
        seed : int, optional
            Random seed.

        Returns
        -------
        ndarray | dict[..]
            The initial observation, check the implementation of `_computeObs()`
        dict[..]
            Additional information as a dictionary, check the implementation of `_computeInfo()`
        """

        # resetSimulation 将从世界中移除所有对象并将世界重置为初始条件。
        p.resetSimulation(physicsClientId=self.physics_client)
        #### Housekeeping ##########################################
        self._housekeeping(self.physics_client)
        #### Update and store the drones kinematic information #####
        self._updateKinematic()
        #### Return the initial observation ########################
        initial_obs = self._computeObs()
        initial_info = self._computeInfo()
        return initial_obs, initial_info

    def close(self):
        self.mav.close()

    def _actionSpace(self):
        """Returns the action space of the environment.
        Returns
        -------
        (right_stroke_amp,
         right_stroke_vel,
         right_input_torque,
         left_stroke_amp,
         left_stroke_vel,
         left_input_torque) = action
        
        if right_stroke_amp is not None and left_stroke_amp is not None and right_stroke_vel is None and left_stroke_vel is None:
        """

        return spaces.Box(
                        low = -1,
                        high = 1,
                        shape = (2,),
                        dtype = np.float32
                        )



    def _observationSpace(self):
        """Returns the observation space of the environment.
        Returns
        -------
        ndarray

        """
        (right_stroke_amp, right_stroke_vel, right_stroke_acc, right_torque,
         left_stroke_amp, left_stroke_vel, left_stroke_acc, left_torque) = self.mav.get_state_for_motor_torque()
        #### Observation vector   ### X        Y        Z        R       P       Y       VX       VY       VZ       WX       WY       WZ       rA        lA
        # obs_lower_bound = np.array([-1,      -1,      0,      -1,     -1,     -1,     -1,      -1,      -1,      -1,      -1,      -1,       -1,      -1])
        # obs_upper_bound = np.array([1,       1,       1,       1,      1,      1,      1,       1,       1,       1,       1,       1,        1,       1])
        return spaces.Box(low=np.array([    -1,-1,0, -1,-1,-1, -1,-1,-1, -1,-1,-1,      -1,-1]),
                              high=np.array([1,1,1,   1,1,1,    1,1,1,     1,1,1,         1,1]),
                              dtype=np.float32
                              )


    def _updateKinematic(self):
        self.pos, self.quat = p.getBasePositionAndOrientation(self.mav.body_unique_id, physicsClientId=self.physics_client)
        self.rpy = p.getEulerFromQuaternion(self.quat)
        self.vel, self.ang_v = p.getBaseVelocity(self.mav.body_unique_id, physicsClientId=self.physics_client)


    def _computeObs(self):

        """Returns the current observation of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (14,).

        """    
        obs = self._clipAndNormalizeState(self._getDroneStateVector())
            # return obs
            ############################################################
            #### OBS SPACE OF SIZE 14
            # XYZ, RPY, V, W，A
        ret = obs[:].reshape(14,)
        return ret.astype('float32')
    

    def _getDroneStateVector(self):
        (right_stroke_amp, right_stroke_vel, right_stroke_acc, right_torque,
         left_stroke_amp, left_stroke_vel, left_stroke_acc, left_torque) = self.mav.get_state_for_motor_torque()
        
        state = np.hstack((self.pos[:], self.rpy[:],
        self.vel[:], self.ang_v[:], right_stroke_amp, left_stroke_amp))
        return state.reshape(14,)


    def _clipAndNormalizeState(self,
                               state):
    
        """Normalizes a hummingbird's state to the [-1,1] range.

        Parameters
        ----------
        state : ndarray
            (20,)-shaped array of floats containing the non-normalized state of a single drone.

        Returns
        -------
        ndarray
            (20,)-shaped array of floats containing the normalized state of a single drone.

        """
        MAX_LIN_VEL_XY = 3
        MAX_LIN_VEL_Z = 3

        MAX_XY = MAX_LIN_VEL_XY*self.EPISODE_LEN_SEC
        MAX_Z = MAX_LIN_VEL_Z*self.EPISODE_LEN_SEC

        MAX_PITCH_ROLL = np.pi # Full range

        clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
        clipped_pos_z = np.clip(state[2], 0, MAX_Z)
        clipped_rp = np.clip(state[3:5], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        clipped_vel_xy = np.clip(state[6:8], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[8], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)

        # Print a warning if values in a state vector is out of the clipping range 
        if not(clipped_pos_xy == np.array(state[0:2])).all():
            print("[WARNING] it", self.step_counter, "in HoverAviary._clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(state[0], state[1]))
        if not(clipped_pos_z == np.array(state[2])).all():
            print("[WARNING] it", self.step_counter, "in HoverAviary._clipAndNormalizeState(), clipped z position [{:.2f}]".format(state[2]))
        if not(clipped_rp == np.array(state[3:5])).all():
            print("[WARNING] it", self.step_counter, "in HoverAviary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(state[3], state[4]))
        if not(clipped_vel_xy == np.array(state[6:8])).all():
            print("[WARNING] it", self.step_counter, "in HoverAviary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(state[6], state[7]))
        if not(clipped_vel_z == np.array(state[8])).all():
            print("[WARNING] it", self.step_counter, "in HoverAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[8]))

        normalized_pos_xy = clipped_pos_xy / MAX_XY
        normalized_pos_z = clipped_pos_z / MAX_Z
        normalized_rp = clipped_rp / MAX_PITCH_ROLL
        normalized_y = state[5] / np.pi # No reason to clip
        normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_XY
        normalized_ang_vel = state[9:12]/np.linalg.norm(state[9:12]) if np.linalg.norm(state[9:12]) != 0 else state[9:12]

        clip_and_norm = np.hstack([normalized_pos_xy,
                                      normalized_pos_z,
                                      normalized_rp,
                                      normalized_y,
                                      normalized_vel_xy,
                                      normalized_vel_z,
                                      normalized_ang_vel,
                                      state[12:14]
                                      ]).reshape(14,)

        return clip_and_norm
    

    def _preAction( self,
                    action
                    ):
        
        return np.array(self.max_angle_of_stroke * action)
        
    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        state = self._getDroneStateVector()
        return -1 * np.linalg.norm(np.array([0, 0, 1])-state[0:3])**2

    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        #   如果   步数（次）*每步周期（秒） >  一个训练episode的时长限制
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False
    
    def _computeTruncated(self):
        """Computes the current truncated value(s).

        Unused in this implementation.

        Returns
        -------
        bool
            Always false.

        """
        return False

    def _computeInfo(self):
        """
        Computes the current info dict(s).
        Unused.
        Returns
        -------
        dict[str, int]
            Dummy value.
        """
        return {"room": 1617}


    def _housekeeping(self,physics_client):

        urdf_creator = URDFCreator(gear_ratio=self.motor_params.gear_ratio,
                                   r=self.wing_params.length,
                                   chord_root=self.wing_params.chord_root,
                                   chord_tip=self.wing_params.chord_tip)
        urdf_name = urdf_creator.write_the_urdf()


        self.mav = BaseRlMAV(urdf_name=urdf_name,
                           mav_params=self.mav_params,
                           client=physics_client,
                           if_fixed=False)

        self.right_motor = BaseBLDC(self.motor_params)
        self.left_motor = BaseBLDC(self.motor_params)
        self.right_wing = BaseWing(self.wing_params)
        self.left_wing = BaseWing(self.wing_params)

        GLOBAL_CONFIGURATION.TICKTOCK = 0
        self.step_counter = 0
        self.mav.housekeeping()
        self.left_motor.housekeeping()
        self.right_motor.housekeeping()
        self.left_wing.housekeeping()
        self.right_wing.housekeeping()

        self.logger = GLOBAL_CONFIGURATION.logger    