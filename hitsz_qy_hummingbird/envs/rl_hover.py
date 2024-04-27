import sys

sys.path.append('/home/hht/simul20240421/240328git/QY-hummingbird/')
from hitsz_qy_hummingbird.wrapper.wrapped_mav_for_RL import RLMAV
from hitsz_qy_hummingbird.configuration import configuration
from hitsz_qy_hummingbird.configuration.configuration import GLOBAL_CONFIGURATION
import numpy as np


class RLhover(RLMAV):
    def __init__(self,
                 urdf_name:str=GLOBAL_CONFIGURATION.temporary_urdf_path + f"Ng_10_AR_5.2_TR_0.8_R22_4e-06.urdf",
                 mav_params=configuration.ParamsForMAV_rl,
                 motor_params=configuration.ParamsForMaxonSpeed6M_rl,
                 wing_params=configuration.ParamsForWing_rl,               
                 gui=False):
        super().__init__(urdf_name, mav_params, motor_params, wing_params, gui)

    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        kp = 8
        kr = 5
        kv = 10
        kw = 0.02

        k1 = 5
        k2 = 0.6
        k3 = 0.02

        reward = 0
        state = self._getDroneStateVector()
        pos_e = np.linalg.norm(self.TARGET_POS - state[0:3])
        att_e = np.abs(state[3] )+ np.abs(2*state[4]) + np.abs(state[5])
        vel_ze = np.abs(state[8])
        augv_e = np.linalg.norm(state[9:12] - np.zeros(3))

        height = state[3]
        acc = 0 
        if(height > 1.2):
            acc= 500

        if  pos_e>=0.3:
            #50
            #-50\18
            #-5*(0\0.5)
            #-10*(2.5\2\1\0.5\0.1)
            reward= 50\
                - kp * ((k1 * pos_e) ** 2) \
                - kr * att_e  \
                - kv * (vel_ze**2)           
        elif (pos_e<0.3 and pos_e>=0.1):
            #100
            #-40*(0.3\0.1)
            #-5*(0\0.5)
            ######-5*(150\81\16\5\1.5)
            #-6*(12.25\9\4\2.25\1.21)
            reward= 100 \
                - 5*kp * pos_e \
                - kr * att_e\
                - k2* kv * (1 + vel_ze)**2
        else:
            #125
            #-80*(0.1\0)
            #-5*(0\0.5)
            #-0.3*(150\81\16\5\1.5)
            #-0.05*50
            self.r_area=self.r_area+1
            reward= 150 \
                - 10*kp * pos_e \
                -  kr * att_e \
                - k3*kv * (1+vel_ze)**4\
                -  kw * augv_e\
                + self.r_area/10
            
        reward = reward- acc 
        return reward
   
