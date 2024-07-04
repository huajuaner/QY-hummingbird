'''
Create a gym-like reinforcement learning environment, 
where the interaction between the FWMAV and the environment is expressed in the step() function.
'''
import numpy as np
import pybullet as p
from pybullet_utils import bullet_client
import pybullet_data
import gymnasium as gym
from gymnasium import spaces
import random
import time
from collections import deque

from hitsz_qy_hummingbird.configuration import configuration
from hitsz_qy_hummingbird.base_FWMAV.MAV.base_MAV_parallel import BaseMavParellel
from hitsz_qy_hummingbird.base_FWMAV.MAV.MAV_params import ParamsForBaseMAV
from hitsz_qy_hummingbird.base_FWMAV.motor.base_BLDC import BaseBLDC
from hitsz_qy_hummingbird.base_FWMAV.motor.motor_params import ParamsForBaseMotor
from hitsz_qy_hummingbird.base_FWMAV.wings.base_Wings import BaseWing
from hitsz_qy_hummingbird.base_FWMAV.wings.wing_params import ParamsForBaseWing
from hitsz_qy_hummingbird.configuration.configuration import GLOBAL_CONFIGURATION
from hitsz_qy_hummingbird.utils.create_urdf import URDFCreator
from hitsz_qy_hummingbird.wrapper.base_wrapped_mav import WrappedBaseMAV


class RLMAV(gym.Env,
            WrappedBaseMAV):
    """
    This class is specially defined for the test of hht-RL
    """

    def __init__(self,
                 urdf_name:str,
                 mav_params=configuration.ParamsForMAV_rl,
                 motor_params=configuration.ParamsForMaxonSpeed6M_rl,
                 wing_params=configuration.ParamsForWing_rl,
                 gui=False,
                 pyb=None,
                 client=None,
                 control_frequency=1200,
                 ):

        self.gui = gui
        if pyb==None:
            # Parallel training, where the BulletClient instance has the same API as the pybullet instance.
            if gui:
                self._p = bullet_client.BulletClient(connection_mode=p.GUI)
            else:
                self._p = bullet_client.BulletClient(connection_mode=p.DIRECT)
            self.physics_client = self._p._client

            lightPosition = [0, 0, 20]
            self._p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            self._p.configureDebugVisualizer(self._p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
            self._p.configureDebugVisualizer(self._p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
            self._p.configureDebugVisualizer(self._p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
            self._p.configureDebugVisualizer(lightPosition=lightPosition)
            self._p.setAdditionalSearchPath(pybullet_data.getDataPath())

            # self.draw_a_setpoint_ball()

            self._p.setGravity(0,0,-9.8)

            self._p.loadURDF("plane.urdf")

            # stepSimulation performs all operations, such as collision detection, constraint solving, and integration, in a single forward dynamics simulation step. 
            # The default time step is 1/240 seconds, which can be changed using the setTimeStep or setPhysicsEngineParameter API.
            # The number of solver iterations and the error reduction parameters (erp) for contact, friction, and non-contact joints are related to the time step. 
            # If you change the time step, you may need to adjust these values accordingly, especially the erp values.
            self._p.setTimeStep(1 / GLOBAL_CONFIGURATION.TIMESTEP)
        else:
            self._p = pyb
            self.physics_client=client

        self.flapper_ID = 0

        self.urdf = urdf_name
        self.mav_params = mav_params
        self.motor_params = motor_params
        self.wing_params = wing_params

        #### Create action and observation spaces ##################

        # self.d_voltage_amplitude_max = 4
        # self.differential_voltage_max = 2 # 3
        # self.mean_voltage_max = 3  # 3.5
        # self.split_cycle_max = 0.1  # 0.1
        # self.voltage_amplitude_max = 20

        # self.hover_voltage_amplitude = 10
        self.d_voltage_amplitude_max = 4
        self.differential_voltage_max = 2 # 3
        self.mean_voltage_max = 3  # 3.5
        self.split_cycle_max = 0.1  # 0.1
        self.voltage_amplitude_max = 20 #20

        self.hover_voltage_amplitude = 10
        self.differential_voltage = 0
        self.mean_voltage = 0
        self.split_cycle = 0

        self.r_voltage = 0
        self.l_voltage = 0
        self.right_stroke_amp = 0
        self.left_stroke_amp = 0

       # wing stroke frequency
        self.frequency = 28
        # pyb and control frequency
        self.PYB_FREQ = GLOBAL_CONFIGURATION.TIMESTEP
        self.CTRL_FREQ = control_frequency
        self.PYB_STEPS_PER_CTRL = int(self.PYB_FREQ / self.CTRL_FREQ)

        self.step_counter = 0
        self.ctrlstep = 0
        self.r_area =0

        #Create a buffer for the last 120 steps of actions 
        self.ACTION_BUFFER_SIZE = int(self.CTRL_FREQ//10)
        self.action_buffer = deque(maxlen=self.ACTION_BUFFER_SIZE)
        for i in range(self.ACTION_BUFFER_SIZE):
            self.action_buffer.append(np.zeros(4))
        #Create a obs buffer
        self.OBS_BUFFER_SIZE = 40
        self.obs_buffer = deque(maxlen=self.OBS_BUFFER_SIZE)
        for i in range(self.OBS_BUFFER_SIZE):
            self.obs_buffer.append(np.zeros(12))

        self.action_space = self._actionSpace()
        self.observation_space = self._observationSpace()

        # Limiting the duration of a training episode
        self.EPISODE_LEN_SEC = 15

        self._housekeeping(self._p, None)

    def step(self,
             action):
        """
        
        """
        for _ in range(self.PYB_STEPS_PER_CTRL):
            self.drive_wing(action)
            self.apply_aeroFT()
            self.mav.step()
            # self.draw_xaxis_of_bf()
            self.step_counter = self.step_counter+1

        # self.step_counter = GLOBAL_CONFIGURATION.TICKTOCK # TICKTOCK has been increased n steps in the loop
        self.ctrlstep = self.ctrlstep + 1
        # self.draw_trac()
        self._updateKinematic()
        obs = self._computeObs()
        reward = self._computeReward()
        terminated = self._computeTerminated()
        truncated = self._computeTruncated()
        info = self._computeInfo()
        return obs, reward, terminated, truncated, info

    def prestep(self,action,sleepflag):
        if sleepflag:
            pass
        else:
            self.drive_wing(action)
            self.apply_aeroFT()
            self.step_counter = self.step_counter+1
    def step_after_pyb(self,
             action):
        """
        
        """   
        # self.draw_zaxis_of_bf()
        # self.step_counter = self.step_counter+1
        # self.draw_trac()
        self._updateKinematic()
        obs = self._computeObs()
        reward = self._computeReward()
        terminated = self._computeTerminated()
        truncated = self._computeTruncated()
        info = self._computeInfo()
        return obs, reward, terminated, truncated, info

    def drive_wing(self, action):
        ''' joint_control of wings' motion.'''
        (self.d_voltage_amplitude,
         self.differential_voltage,
         self.mean_voltage,
         self.split_cycle,) = self._preAction(action)

        self.voltage_amplitude = self.hover_voltage_amplitude + self.d_voltage_amplitude

        (self.right_stroke_amp, right_stroke_vel, right_stroke_acc, _,
         self.left_stroke_amp, left_stroke_vel, left_stroke_acc, _) = self.mav.get_state_for_motor_torque()

        self.r_voltage = self.generate_control_signal(self.frequency,
                                                      self.voltage_amplitude,
                                                      -self.differential_voltage,
                                                      self.mean_voltage,
                                                      self.split_cycle,
                                                      self.step_counter / GLOBAL_CONFIGURATION.TIMESTEP,
                                                      0, )
        self.l_voltage = -self.generate_control_signal(self.frequency,
                                                       self.voltage_amplitude,
                                                       self.differential_voltage,
                                                       self.mean_voltage,
                                                       -self.split_cycle,
                                                       self.step_counter / GLOBAL_CONFIGURATION.TIMESTEP,
                                                       0, )
        self.r_voltage = np.clip(self.r_voltage, -self.voltage_amplitude_max, self.voltage_amplitude_max)
        self.l_voltage = np.clip(self.l_voltage, -self.voltage_amplitude_max, self.voltage_amplitude_max)

        r_torque = self.right_motor.step(voltage=self.r_voltage,
                                         stroke_angular_amp=self.right_stroke_amp,
                                         stroke_angular_vel=right_stroke_vel,
                                         stroke_angular_acc=right_stroke_acc,
                                         if_record=False)

        l_torque = self.left_motor.step(voltage=self.l_voltage,
                                        stroke_angular_amp=self.left_stroke_amp,
                                        stroke_angular_vel=left_stroke_vel,
                                        stroke_angular_acc=left_stroke_acc,
                                        if_record=False)

        self.mav.joint_control(target_right_stroke_amp=None,
                               target_right_stroke_vel=None,
                               right_input_torque=r_torque,
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
            linkid=self.mav.params.left_wing_link,
            torque=left_aerotorque
        )

    def draw_xaxis_of_bf(self):
        flapperPos, flapperOrn = self._p.getBasePositionAndOrientation(self.flapper_ID)
        flapperRot = np.array(self._p.getMatrixFromQuaternion(flapperOrn)).reshape(3, 3)
        x_axis = flapperRot[:, 0]
        # print(f"zaxis={zaxis}")
        # print(f"cubePos+0.5*zaxis={cubePos+zaxis}")
        self.mav.draw_a_line(flapperPos, flapperPos + 0.03 * x_axis, [1, 0, 0], f'torso')

    def draw_zaxis_of_bf(self):
        flapperPos, flapperOrn = self._p.getBasePositionAndOrientation(self.flapper_ID)
        flapperRot = np.array(self._p.getMatrixFromQuaternion(flapperOrn)).reshape(3, 3)
        zaxis = flapperRot[:, 2]
        # print(f"zaxis={zaxis}")
        # print(f"cubePos+0.5*zaxis={cubePos+zaxis}")
        self.mav.draw_a_line(flapperPos, flapperPos + 0.03 * zaxis, [1, 0, 0], f'torso')

    def _getotherdata(self):
        return np.array([self.r_voltage, self.l_voltage, self.right_stroke_amp, self.left_stroke_amp])

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

    def reset(self,
              seed: int = None, ):
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

        # self._p.resetSimulation(physicsClientId=self.physics_client)
        self._p.removeBody(self.flapper_ID)
        #### Housekeeping ##########################################
        self._housekeeping(self._p, seed)
        #### Update and store the FWMAV's kinematic information #####
        self._updateKinematic()
        initial_obs = self._computeObs()
        initial_info = self._computeInfo()
        return initial_obs, initial_info

    def close(self):
        self.mav.close()

    def _actionSpace(self):
        """Returns the action space of the environment.
        Returns normalized
        -------
        self.d_voltage_amplitude
        self.differential_voltage
        self.mean_voltage
        self.d_split_cycle
        """
        return spaces.Box(
            low=np.array([-1, -1, -1, -1]),
            high=np.array([1, 1, 1, 1]),
            shape=(4,),
            dtype=np.float32
        )

    def _preAction(self,
                   action
                   ):
        '''store last action and De-normalization'''
        #Inserting at the head of the deque
        self.action_buffer.appendleft(action)
        return (self.d_voltage_amplitude_max * action[0],
                self.differential_voltage_max * action[1],
                self.mean_voltage_max * action[2],
                self.split_cycle_max * action[3])

    def _observationSpace(self):
        """Returns the observation space of the environment.
        Returns
        -------
        ndarray

        """
        # (right_stroke_amp, right_stroke_vel, right_stroke_acc, right_torque,
        #  left_stroke_amp, left_stroke_vel, left_stroke_acc, left_torque) = self.mav.get_state_for_motor_torque()
        #### Observation vector   ### eX        eY        eZ        er       ep       ey       VX       VY       VZ       WX       WY       WZ        U       dU        U0       sc
        # obs_lower_bound = np.array([-1,      -1,      -1,      -1,     -1,     -1,     -1,      -1,      -1,      -1,      -1,      -1,       -1,      -1,       -1,      -1])
        # obs_upper_bound = np.array([1,       1,       1,       1,      1,      1,      1,       1,       1,       1,       1,       1,        1,       1,       -1,      -1])
        # low=np.array([-1,-1, 0, -np.pi,-np.pi/2, -np.pi, -np.inf,-np.inf,-np.inf, -np.inf,-np.inf,-np.inf,])
        # high=np.array([1, 1, 1,  np.pi, np.pi/2,  np.pi,    np.inf,np.inf,np.inf,     np.inf,np.inf,np.inf,])
        obs_low = np.array([-0.5, -0.5, -0.5, -np.pi, -np.pi/2, -np.pi, -5, -5, -5, -100, -100, -100])
        obs_high = np.array([0.5, 0.5, 0.5, np.pi, np.pi/2, np.pi, 5, 5, 5, 100, 100, 100])
        tiled_obs_low = np.tile(obs_low, self.OBS_BUFFER_SIZE)
        tiled_obs_high = np.tile(obs_high, self.OBS_BUFFER_SIZE)

        act_low = np.array([-1, -1, -1, -1])
        act_high = np.array([1, 1, 1, 1])
        tiled_act_low = np.tile(act_low, self.ACTION_BUFFER_SIZE)
        tiled_act_high = np.tile(act_high, self.ACTION_BUFFER_SIZE)

        # Concatenate 
        low = np.hstack([tiled_obs_low, tiled_act_low])
        high = np.hstack([tiled_obs_high, tiled_act_high])

        return spaces.Box(low,
                          high,
                          dtype=np.float32
                          )

    def _updateKinematic(self):
        self.pos, self.quat = self._p.getBasePositionAndOrientation(self.flapper_ID)
        self.rpy = self._p.getEulerFromQuaternion(self.quat)
        self.vel, self.ang_v = self._p.getBaseVelocity(self.flapper_ID)

    def _computeObs(self):

        """Returns the current observation of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (16,).

        """
        cur_obs = self._getDroneStateVector()
        self.obs_buffer.appendleft(cur_obs)

        obs_flat = np.concatenate(self.obs_buffer)
        act_flat = np.concatenate(self.action_buffer)
        obs = np.concatenate([obs_flat, act_flat])
        ret = obs[:].reshape((self.OBS_BUFFER_SIZE*12+self.ACTION_BUFFER_SIZE*4), )
        return ret.astype('float32')

    def _getDroneStateVector(self):
        # OBS SPACE OF SIZE 12
        # XYZ, rpy, V, W
        pos = self.pos[:]
        rpy = self.rpy[:]
        state = np.hstack((pos[:], rpy[:],
                           self.vel[:], self.ang_v[:]))
        return state.reshape(12, )

    def _clipAndNormalizeState(self,
                               state):

        """Normalizes a FWMAV's state to the [-1,1] range.

        Parameters
        ----------
        state : ndarray
            (16,)-shaped array of floats containing the non-normalized state of a single FWMAV.

        Returns
        -------
        ndarray
            (16,)-shaped array of floats containing the normalized state of a single FWMAV.

        """
        MAX_LIN_VEL_XY = 5
        MAX_LIN_VEL_Z = 10

        MAX_XY = 2
        MAX_Z = 2

        MAX_ROLL = np.pi  # Full range
        MAX_PITCH = np.pi / 2

        clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
        clipped_pos_z = np.clip(state[2], 0, MAX_Z)
        clipped_roll = np.clip(state[3], -MAX_ROLL, MAX_ROLL)
        clipped_pitch = np.clip(state[4], -MAX_PITCH, MAX_PITCH)
        clipped_vel_xy = np.clip(state[6:8], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[8], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)

        # Print a warning if values in a state vector is out of the clipping range 
        # if not(clipped_pos_xy == np.array(state[0:2])).all():
        #    print("[WARNING] it", self.step_counter, "in HoverAviary._clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(state[0], state[1]))
        # if not(clipped_pos_z == np.array(state[2])).all():
        #    print("[WARNING] it", self.step_counter, "in HoverAviary._clipAndNormalizeState(), clipped z position [{:.2f}]".format(state[2]))
        # if not(clipped_rp == np.array(state[3:5])).all():
        #    print("[WARNING] it", self.step_counter, "in HoverAviary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(state[3], state[4]))
        # if not(clipped_vel_xy == np.array(state[6:8])).all():
        #    print("[WARNING] it", self.step_counter, "in HoverAviary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(state[6], state[7]))
        # if not(clipped_vel_z == np.array(state[8])).all():
        #    print("[WARNING] it", self.step_counter, "in HoverAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[8]))

        normalized_pos_xy = clipped_pos_xy / MAX_XY
        normalized_pos_z = clipped_pos_z / MAX_Z
        normalized_roll = clipped_roll / MAX_ROLL
        normalized_pitch = clipped_pitch / MAX_PITCH
        normalized_yaw = state[5] / np.pi
        normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_Z
        normalized_ang_vel = state[9:12] / np.linalg.norm(state[9:12]) if np.linalg.norm(state[9:12]) != 0 else state[
                                                                                                                9:12]

        normalized_d_voltage_amplitude = state[12]
        normalized_differential_voltage = state[13]
        normalized_mean_voltage = state[14]
        normalized_split_cycle = state[15]

        clip_and_norm = np.hstack([normalized_pos_xy,
                                   normalized_pos_z,
                                   normalized_roll,
                                   normalized_pitch,
                                   normalized_yaw,
                                   normalized_vel_xy,
                                   normalized_vel_z,
                                   normalized_ang_vel,
                                   normalized_d_voltage_amplitude,
                                   normalized_differential_voltage,
                                   normalized_mean_voltage,
                                   normalized_split_cycle
                                   ]).reshape(16, )

        return clip_and_norm

    def _computeReward(self):
        """

        Each subclass has its own reward function.

        """
        raise NotImplementedError

    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """

        # state = self._getDroneStateVector()
        # if np.linalg.norm(state[0:3]) < .0001 and state[8]<.0001:
        #     return True
        # else:
        return False

    def _computeTruncated(self):
        """Computes the current truncated value(s).
        """
        state = self._getDroneStateVector()
        #  If flying too far
        if (abs(state[0]) > 0.3 or abs(state[1]) > 0.3 or abs(state[2]) > 0.3
                or abs(state[3]) > np.pi / 4 or abs(state[4]) > np.pi / 4 or abs(state[5]) > np.pi / 4):
            return True
        # Maintain vertical attitude
        Rot = np.array(self._p.getMatrixFromQuaternion(self.quat)).reshape(3, 3)
        theta_z = np.arccos(Rot[2, 2])
        if (theta_z > np.pi / 6):
            return True
        #   If the number of pyb steps * the duration of each step (seconds) 
        #   > the duration limit of one training episode
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
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

    def _housekeeping(self, p_this, seed0):
        # Create MAV
        self.mav = BaseMavParellel(
            urdf_name= self.urdf,
            mav_params=self.mav_params,
            pyb=p_this,
            if_gui=self.gui,
            if_fixed=False)
        self.flapper_ID = self.mav.body_unique_id

        self.right_motor = BaseBLDC(self.motor_params)
        self.left_motor = BaseBLDC(self.motor_params)
        self.right_wing = BaseWing(self.wing_params)
        self.left_wing = BaseWing(self.wing_params)

        GLOBAL_CONFIGURATION.TICKTOCK = 0
        self.step_counter = 0
        self.ctrlstep =0
        self.r_area =0

        # data clear
        self.mav.housekeeping()
        self.left_motor.housekeeping()
        self.right_motor.housekeeping()
        self.left_wing.housekeeping()
        self.right_wing.housekeeping()

        # self.logger = GLOBAL_CONFIGURATION.logger

    def draw_trac(self):
        flapperPos,_= self._p.getBasePositionAndOrientation(self.flapper_ID,
                                                    physicsClientId=self.physics_client)
        self._p.addUserDebugLine(flapperPos, flapperPos + 0.001*np.array([0., 0., 1.0]),[1, 0, 0])

    def erase_trac(self):
        self._p.removeAllUserDebugItems()

    def draw_a_setpoint_ball(self):
        # 定义球的位置和半径
        sphere_radius = 0.005
        sphere_position = [1, 0, 0.5]

        # 在给定位置创建一个纯视觉球体
        visual_shape_id = self._p.createVisualShape(shapeType=p.GEOM_SPHERE,
                                            radius=sphere_radius,
                                            rgbaColor=[1, 1, 0, 0.7], 
                                            specularColor=[1, 1, 1, 1])

        # 由于不需要碰撞形状，直接创建多体只使用视觉形状
        sphere_body_id = self._p.createMultiBody(baseMass=0,
                                        baseVisualShapeIndex=visual_shape_id,
                                        basePosition=sphere_position)

    def pause(self):
        for i in range(self.PYB_STEPS_PER_CTRL):
            self.mav.pause()