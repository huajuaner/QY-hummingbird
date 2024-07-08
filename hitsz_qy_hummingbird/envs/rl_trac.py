import sys

sys.path.append('/home/hht/simul0703/QY-hummingbird/')
from hitsz_qy_hummingbird.wrapper.wrapped_mav_for_RL import RLMAV
from hitsz_qy_hummingbird.configuration import configuration
from hitsz_qy_hummingbird.base_FWMAV.MAV.MAV_params import ParamsForBaseMAV
from hitsz_qy_hummingbird.configuration.configuration import GLOBAL_CONFIGURATION
import numpy as np
import copy

class RLtrac(RLMAV):
    def __init__(self,
                 urdf_name:str=GLOBAL_CONFIGURATION.temporary_urdf_path + f"Ng_10_AR_5.2_TR_0.8_R22_4e-06.urdf",
                 initial_pos=np.array([0, 0, 0.5]),
                 initial_att=np.array([0, 0, 0]),
                 target_pos=np.array([0, 0, 0.5]),
                 target_att=np.array([0, 0, 0]),
                 trunc_flag=True,
                 mav_params=configuration.ParamsForMAV_rl,
                 motor_params=configuration.ParamsForMaxonSpeed6M_rl,
                 wing_params=configuration.ParamsForWing_rl,               
                 gui=False,
                 pyb=None,
                 client=None,):
        mav_params_copy = copy.deepcopy(mav_params)
        mav_params_copy.change_parameters(initial_xyz=initial_pos)
        mav_params_copy.change_parameters(initial_rpy=initial_att)
        self.TARGET_POS = target_pos
        self.TARGET_RPY = target_att
        self.trunc_flag = trunc_flag
        self.attzero_flag = True
        super().__init__(urdf_name, mav_params_copy, motor_params, wing_params, gui, pyb, client)
    
    def step(self,
             action,
             trac_pos):
        """
        
        """
        for _ in range(self.PYB_STEPS_PER_CTRL):
            self.drive_wing(action)
            self.apply_aeroFT()
            self.mav.step()
            self.draw_zaxis_of_bf()
            self.step_counter = self.step_counter+1

        # self.step_counter = GLOBAL_CONFIGURATION.TICKTOCK # TICKTOCK has been increased n steps in the loop
        self.ctrlstep = self.ctrlstep + 1
        self.draw_trac()
        self._updateKinematic()
        obs = self._computeObs(trac_pos)
        reward = self._computeReward()
        terminated = self._computeTerminated()
        truncated = self._computeTruncated()
        info = self._computeInfo()
        return obs, reward, terminated, truncated, info
    
    def reset(self,
              trac_pos,
              seed: int = None,):
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
        #### Housekeeping ##########################################
        self._housekeeping(self._p, seed)
        #### Update and store the FWMAV's kinematic information #####
        self._updateKinematic()
        initial_obs = self._computeObs(trac_pos)
        initial_info = self._computeInfo()
        return initial_obs, initial_info
    
    def _computeObs(self,trac_pos):

        """Returns the current observation of the environment.

        Returns
        -------
        ndarrayself.TARGET_POS
            A Box() of shape (16,).

        """
        cur_obs = self._getDroneStateVector(trac_pos)
        self.obs_buffer.appendleft(cur_obs)

        obs_flat = np.concatenate(self.obs_buffer)
        act_flat = np.concatenate(self.action_buffer)
        obs = np.concatenate([obs_flat, act_flat])
        ret = obs[:].reshape((self.OBS_BUFFER_SIZE*12+self.ACTION_BUFFER_SIZE*4), )
        return ret.astype('float32')

    def _getDroneStateVector(self,trac_pos):
        self.TARGET_POS = trac_pos 
        # OBS SPACE OF SIZE 12
        # eXYZ, erpy, V, W
        # temporarily assume that target_att approx initial_att
        if self.attzero_flag:
            err_pos = self.pos[:]-self.TARGET_POS[:]
            err_rpy = self.rpy[:]-self.TARGET_RPY[:]
            state = np.hstack((err_pos[:], err_rpy[:],
                            self.vel[:], self.ang_v[:]))
            return state.reshape(12, )
        else:
            #e_代表地面坐标系,t_代表按初始姿态转动了一定yaw角的地面坐标系,b_代表机体坐标系
            err_e_pos = np.array(self.pos[:]-self.TARGET_POS[:])
            e_vel =  np.array(self.vel[:])
            e_ang_v = np.array(self.ang_v[:])
            bRe = np.array(self._p.getMatrixFromQuaternion(self.quat)).reshape(3, 3)

            # tQe = self._p.getQuaternionFromEuler(self.TARGET_RPY)
            # tRe = np.array(self._p.getMatrixFromQuaternion(tQe)).reshape(3, 3)
            # 更通用地应该用tRe求逆矩阵,得到eRt,下面只考虑了yaw的旋转
            eQt = self._p.getQuaternionFromEuler(-self.TARGET_RPY)
            eRt = np.array(self._p.getMatrixFromQuaternion(eQt)).reshape(3, 3)

            # Convert e_pos, bRe, e_vel, and e_ang_v to column vectors
            err_e_pos_col = err_e_pos.reshape(-1, 1)
            e_vel_col = e_vel.reshape(-1, 1)
            e_ang_v_col = e_ang_v.reshape(-1, 1)

            # Stack column vectors to form a 3x6 matrix
            matrix_3x6 = np.hstack((err_e_pos_col, bRe, e_vel_col, e_ang_v_col))

            # Perform left multiplication of eRt with the 3x6 matrix
            result_matrix = np.dot(eRt, matrix_3x6)

            # Extract individual columns from the resulting matrix
            err_t_pos = result_matrix[:, 0]
            bRt = result_matrix[:, 1:4].reshape(3, 3)
            t_quat = self._getQuaternionFromMatrix(bRt)
            # 使用 PyBullet 的 getEulerFromQuaternion 函数获取欧拉角
            t_att = self._p.getEulerFromQuaternion(t_quat)
            t_vel = result_matrix[:, 4]
            t_ang_v = result_matrix[:, 5]

            state = np.hstack((err_t_pos[:], t_att[:],
                            t_vel[:], t_ang_v[:]))
            return state.reshape(12, )
        
    def _gettrue_xyzrpy(self):
        pos = self.pos[:]
        rpy = self.rpy[:]
        return np.hstack((pos,rpy)).reshape(6,)

    def _getQuaternionFromMatrix(self,bRe):
        # 假设您已经有一个旋转矩阵 bRe
        bRe = np.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]])

        # 计算四元数
        w = np.sqrt(1 + bRe[0, 0] + bRe[1, 1] + bRe[2, 2]) / 2
        x = (bRe[2, 1] - bRe[1, 2]) / (4 * w)
        y = (bRe[0, 2] - bRe[2, 0]) / (4 * w)
        z = (bRe[1, 0] - bRe[0, 1]) / (4 * w)

        quaternion = [x, y, z, w]
        return quaternion
    
    def add_debug_circle_xy(self):
        # 圆心
        center = [0, 0, 0.5]

        # 半径
        radius = 0.2

        # 绘制圆
        num_points = 32  # 使用的点的数量，越大越接近圆形
        for i in range(num_points):
            theta = 2 * np.pi * i / num_points
            next_theta = 2 * np.pi * (i + 1) / num_points
            x1 = center[0] + radius * np.cos(theta)
            y1 = center[1] + radius * np.sin(theta)
            x2 = center[0] + radius * np.cos(next_theta)
            y2 = center[1] + radius * np.sin(next_theta)
            self._p.addUserDebugLine([x1, y1, center[2]], [x2, y2, center[2]], [0, 0, 1], 3)

    def add_debug_circle_xz(self):
        # 圆心
        center = [0, 0, 0.3]

        # 半径
        radius = 0.2

        # 绘制圆
        num_points = 32  # 使用的点的数量，越大越接近圆形
        for i in range(num_points):
            theta = 2 * np.pi * i / num_points
            next_theta = 2 * np.pi * (i + 1) / num_points
            x1 = center[0] + radius * np.cos(theta)
            z1 = center[2] + radius * np.sin(theta)
            x2 = center[0] + radius * np.cos(next_theta)
            z2 = center[2] + radius * np.sin(next_theta)
            self._p.addUserDebugLine([x1, center[1], z1], [x2, center[1], z2], [0, 0, 1], 3)

    def _computeTruncated(self):
        """Computes the current truncated value(s).
        """
        return False


    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        return 1
