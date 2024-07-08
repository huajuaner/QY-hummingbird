import sys

sys.path.append('/home/hht/simul0703/QY-hummingbird/')
from hitsz_qy_hummingbird.wrapper.wrapped_mav_for_RL import RLMAV
from hitsz_qy_hummingbird.configuration import configuration
from hitsz_qy_hummingbird.base_FWMAV.MAV.MAV_params import ParamsForBaseMAV
from hitsz_qy_hummingbird.configuration.configuration import GLOBAL_CONFIGURATION
import numpy as np
import copy
import random

class RLatt(RLMAV):
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
        self.target_rpy0_flag = np.array_equal(self.TARGET_RPY, np.array([0, 0, 0]))
        self.trunc_flag = trunc_flag
        super().__init__(urdf_name, mav_params_copy, motor_params, wing_params, gui, pyb, client)

    def _getDroneStateVector(self):
        # OBS SPACE OF SIZE 12
        # eXYZ, erpy, V, W
        # temporarily assume that target_att approx initial_att
        if self.target_rpy0_flag:
            err_pos = self.pos[:]-self.TARGET_POS[:]
            err_rpy = self.rpy[:]-self.TARGET_RPY[:]
            state = np.hstack((err_pos[:], err_rpy[:],
                            self.vel[:], self.ang_v[:]))
            return state.reshape(12, )
        else:
            #aRb means the rotation matrix of a in b
            #e_ represents the ground coordinate system, 
            #t_ represents the ground coordinate system rotated by a certain yaw angle from the initial attitude,
            #b_ represents the body coordinate system.
            err_e_pos = np.array(self.pos[:]-self.TARGET_POS[:])
            e_vel =  np.array(self.vel[:])
            e_ang_v = np.array(self.ang_v[:])
            bRe = np.array(self._p.getMatrixFromQuaternion(self.quat)).reshape(3, 3)

            # tQe = self._p.getQuaternionFromEuler(self.TARGET_RPY)
            # tRe = np.array(self._p.getMatrixFromQuaternion(tQe)).reshape(3, 3)
            # More generally, should use the transpose of the tRe​ matrix to obtain the eRt​ matrix. 
            # Below, only the rotation about the yaw is considered.
            eQt = self._p.getQuaternionFromEuler(-self.TARGET_RPY)
            eRt = np.array(self._p.getMatrixFromQuaternion(eQt)).reshape(3, 3)

            # Convert e_pos, e_vel, and e_ang_v to column vectors
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
            t_att = self._p.getEulerFromQuaternion(t_quat)

            t_vel = result_matrix[:, 4]
            t_ang_v = result_matrix[:, 5]

            state = np.hstack((err_t_pos[:], t_att[:],
                            t_vel[:], t_ang_v[:]))
            return state.reshape(12, )
        
    def _getQuaternionFromMatrix(self,bRe):
        
        bRe = np.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]])

        # quaternion
        w = np.sqrt(1 + bRe[0, 0] + bRe[1, 1] + bRe[2, 2]) / 2
        x = (bRe[2, 1] - bRe[1, 2]) / (4 * w)
        y = (bRe[0, 2] - bRe[2, 0]) / (4 * w)
        z = (bRe[1, 0] - bRe[0, 1]) / (4 * w)

        quaternion = [x, y, z, w]
        return quaternion
    
    def _gettrue_xyzrpy(self):
        pos = self.pos[:]
        rpy = self.rpy[:]
        return np.hstack((pos,rpy)).reshape(6,)

    def _computeTruncated(self):
        """Computes the current truncated value(s).
        """
        if self.trunc_flag:
            state = self._getDroneStateVector()
            #  If flying too far
            if (abs(state[0]) > 0.05 or abs(state[1]) > 0.05 or abs(state[2]) >0.3 or
                abs(state[3]) > np.pi / 4 or abs(state[4]) > np.pi / 4 or abs(state[5]) > np.pi / 4):
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
        else:
            return False

    def _housekeeping(self, p_this, seed0):

        # Randomize initial position and attitude
        # if seed0:
        #     seed = seed0
        # else:
        #     seed = random.randint(0, 1000)
        # z_position = np.random.uniform(0.4, 0.6)
        # random_pos = np.array([0, 0, z_position])
        # np.random.seed(seed)
        # random_att = np.pi / 72 * (2 * np.random.rand(3) - [1, 1, 1])
        # self.mav_params.change_parameters(initial_xyz=random_pos)
        # self.mav_params.change_parameters(initial_rpy=random_att)
        super()._housekeeping(p_this, seed0)

    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        kpz = 8
        kpxy = 200
        kr = 40
        kvy = 20
        kvz = 5
        kw = 0.1

        reward = 0
        state = self._getDroneStateVector()
        pos_z_e = np.abs(state[2] )
        pos_xy_e = np.linalg.norm(state[0:2])
        pos_e = np.linalg.norm(state[0:3])
        att_e = np.abs(state[3] )+ np.abs(state[4]) + np.abs(state[5])
        vel_y_e = np.abs(state[7])
        vel_z_e = np.abs(state[8])
        w_e = np.abs(state[9])+np.abs(state[10])+np.abs(state[11])

        #It's difficult to learn punishment for vxvx​ and vyvy​ because vxvx​ is generated through pitch, 
        #and vyvy​ is generated through yaw and pitch (the vyvy​ generated by roll is relatively small)
        #25
        #-200*(0.15\0.05\0.02\0)
        #-8*(1\0.25\0)<--(0.1\0.05\0)
        #-40*(0\0.25)
        #-20*(1.44\1.21)<--(0.2\0.1)
        #-5*(2.25\1.44\1.21)<--(0.5\0.2\0.1)
        #-0.1*50

        reward= 50\
                - kpxy *pos_xy_e\
                - kpz *(10*pos_z_e)**2 \
                - kr * att_e  \
                - kvy * ((1+vel_y_e) ** 2) \
                - kvz * ((1+vel_z_e) ** 2)  \
                - kw * w_e 
        
        if(pos_e<0.025):
                self.r_area=self.r_area+1
                reward = reward+ 5+self.r_area/1000
            
        return reward