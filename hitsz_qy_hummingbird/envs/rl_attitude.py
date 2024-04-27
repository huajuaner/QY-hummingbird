import sys

sys.path.append('/home/hht/simul20240421/240328git/QY-hummingbird/')
from hitsz_qy_hummingbird.wrapper.wrapped_mav_for_RL import RLMAV
from hitsz_qy_hummingbird.configuration import configuration

from hitsz_qy_hummingbird.configuration.configuration import GLOBAL_CONFIGURATION
import numpy as np

class RLatt(RLMAV):
    def __init__(self,
                 urdf_name:str=GLOBAL_CONFIGURATION.temporary_urdf_path + f"Ng_10_AR_5.2_TR_0.8_R22_4e-06.urdf",
                 mav_params=configuration.ParamsForMAV_rl,
                 motor_params=configuration.ParamsForMaxonSpeed6M_rl,
                 wing_params=configuration.ParamsForWing_rl,               
                 gui=False):
        self.TARGET_POS = np.array([0, 0, 0.5])
        self.TARGET_RPY = np.array([0, 0, 0])
        super().__init__(urdf_name, mav_params, motor_params, wing_params, gui)


    def _computeTruncated(self):
        """Computes the current truncated value(s).
        """
        state = self._getDroneStateVector()
        #  If flying too far
        if (abs(state[0]) > 0.1 or abs(state[1]) > 0.1 or abs(state[2]) >0.3 
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


    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """

        kpz = 8
        kpxy = 100
        kr = 40
        kvz = 5
        kw = 0.1

        reward = 0
        state = self._getDroneStateVector()
        pos_z_e = np.abs(state[2] )
        pos_xy_e = np.linalg.norm(state[0:2])
        pos_e = np.linalg.norm(state[0:3])
        att_e = np.abs(state[3] )+ np.abs(state[4]) + 2*np.abs(state[5])
        vel_z_e = np.abs(state[8])
        w_e = np.abs(state[9])+np.abs(state[10])+4*np.abs(state[11])

        #对vx,vy惩罚学不到，因为是通过俯仰产生vx，偏航+俯仰产生vy(滚转产生的vy较小)
        #25
        #-100*(0.15\0.05\0.02\0)
        #-8*(1\0.25\0)<--(0.1\0.05\0)
        #-40*(0\0.25)
        #-5*(12.25\9\4\2.25\1.21)
        #-0.1*50

        reward= 25\
                - kpxy *pos_xy_e\
                - kpz *(10*pos_z_e)**2 \
                - kr * att_e  \
                - kvz * ((1+vel_z_e) ** 2)  \
                - kw * w_e 
        
        if(pos_e<0.025):
                reward = reward+ 10
         
        return reward

