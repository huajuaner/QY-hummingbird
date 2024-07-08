import sys

sys.path.append('/home/hht/simul0703/QY-hummingbird/')
from hitsz_qy_hummingbird.wrapper.wrapped_mav_for_RL import RLMAV
from hitsz_qy_hummingbird.configuration import configuration
from hitsz_qy_hummingbird.base_FWMAV.MAV.MAV_params import ParamsForBaseMAV
from hitsz_qy_hummingbird.configuration.configuration import GLOBAL_CONFIGURATION
import numpy as np
import copy
import random

class RLattbody(RLMAV):
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
        #aRb means the rotation matrix of a in b
        #e_ represents the ground coordinate system, 
        #b_ represents the body coordinate system.
        e_err_pos = self.pos[:]-self.TARGET_POS[:]
        e_err_rpy = self.rpy[:]-self.TARGET_RPY[:]
        e_vel_col = np.array(self.vel[:]).reshape(-1,1)
        e_ang_v_col = np.array(self.ang_v[:]).reshape(-1,1)
        matrix_3x2 = np.hstack((e_vel_col,e_ang_v_col))
        bRe = np.array(self._p.getMatrixFromQuaternion(self.quat)).reshape(3, 3)
        eRb = bRe.T
        result_matrix = np.dot(eRb, matrix_3x2)
        b_vel = result_matrix[:, 0]
        b_ang_v = result_matrix[:, 1]
        state = np.hstack((e_err_pos[:], e_err_rpy[:],
                        b_vel[:], b_ang_v[:]))
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
            if (abs(state[0]) > 0.1 or abs(state[1]) > 0.1 or abs(state[2]) >0.3 or
                abs(state[3]) > np.pi / 4 or abs(state[4]) > np.pi / 4 ):
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
        if seed0:
            seed = seed0
        else:
            seed = random.randint(0, 1000)
        z_position = np.random.uniform(0.4, 0.6)
        random_pos = np.array([0, 0, z_position])
        np.random.seed(seed)
        roll = np.pi / 72 * (2 * np.random.rand(1) - 1)
        pitch = np.pi / 72 * (2 * np.random.rand(1) - 1)
        yaw = np.pi * (2 * np.random.rand(1) - 1)
        random_att = np.array([roll.item(), pitch.item(), yaw.item()])
        self.mav_params.change_parameters(initial_xyz=random_pos)
        self.mav_params.change_parameters(initial_rpy=random_att)
        self.TARGET_RPY = np.array([0, 0, yaw.item()])
        super()._housekeeping(p_this, seed0)

    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        #0.99,256
        kp =8
        kr = 10
        kvz = 5
        kw = 0.1
        k1=10
        reward = 0
        state = self._getDroneStateVector()
        pos_e = np.linalg.norm(state[0:3])
        att_e = np.abs(state[3] )+ 4*np.abs(state[4]) + np.abs(state[5])
        vel_z_e = np.abs(state[8])
        w_e = np.abs(state[9])+np.abs(state[10])+ np.abs(state[11])

        #25
        #-8*(1\0.25\0)<--(0.1\0.05\0)
        #-10*(0\0.25)
        #-5*(2.25\1.21)<--(0.2\0.1)
        #-0.1*50

        reward= 25\
                - kp *((k1*pos_e)**2)\
                - kr * att_e  \
                - kvz * ((1+vel_z_e) ** 2)  \
                - kw * w_e 
        
        if(pos_e<0.025):
                self.r_area=self.r_area+1
                reward = reward+ 5+self.r_area/100
         
        return reward
        # kpz = 8
        # kpxy = 200
        # kr = 40
        # kvy = 20
        # kvz = 5
        # kw = 0.1

        # reward = 0
        # state = self._getDroneStateVector()
        # pos_z_e = np.abs(state[2] )
        # pos_xy_e = np.linalg.norm(state[0:2])
        # pos_e = np.linalg.norm(state[0:3])
        # att_e = np.abs(state[3] )+ np.abs(state[4]) + np.abs(state[5])
        # vel_y_e = np.abs(state[7])
        # vel_z_e = np.abs(state[8])
        # w_e = np.abs(state[9])+np.abs(state[10])+np.abs(state[11])

        # #It's difficult to learn punishment for vxvx​ and vyvy​ because vxvx​ is generated through pitch, 
        # #and vyvy​ is generated through yaw and pitch (the vyvy​ generated by roll is relatively small)
        # #25
        # #-200*(0.15\0.05\0.02\0)
        # #-8*(1\0.25\0)<--(0.1\0.05\0)
        # #-40*(0\0.25)
        # #-20*(1.44\1.21)<--(0.2\0.1)
        # #-5*(2.25\1.44\1.21)<--(0.5\0.2\0.1)
        # #-0.1*50

        # reward= 50\
        #         - kpxy *pos_xy_e\
        #         - kpz *(10*pos_z_e)**2 \
        #         - kr * att_e  \
        #         - kvy * ((1+vel_y_e) ** 2) \
        #         - kvz * ((1+vel_z_e) ** 2)  \
        #         - kw * w_e 
        
        # if(pos_e<0.025):
        #         self.r_area=self.r_area+1
        #         reward = reward+ 5+self.r_area/1000
            
        # return reward