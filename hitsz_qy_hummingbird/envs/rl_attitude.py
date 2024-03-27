import sys
from hitsz_qy_hummingbird.configuration.configuration import ParamsForMAV_rl, ParamsForMaxonSpeed6M_rl, ParamsForWing_rl 
sys.path.append('D://graduate//fwmav//simul2024//240325git//QY-hummingbird')
from hitsz_qy_hummingbird.envs.RL_wrapped import RLMAV
from hitsz_qy_hummingbird.configuration import configuration


class RLatt(RLMAV):
    def __init__(self,
                    mav_params=configuration.ParamsForMAV_rl, 
                    motor_params=configuration.ParamsForMaxonSpeed6M_rl,
                    wing_params=configuration.ParamsForWing_rl, 
                    gui=False):
        super().__init__(mav_params, motor_params, wing_params, gui)

    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        kp = 7
        kr = 3
        state = self._getDroneStateVector()
        return 0  