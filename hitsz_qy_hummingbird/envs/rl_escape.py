import sys

sys.path.append('/home/hht/simul0703/QY-hummingbird/')
from hitsz_qy_hummingbird.wrapper.wrapped_mav_for_RL import RLMAV
from hitsz_qy_hummingbird.configuration import configuration


class RLescape(RLMAV):
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
        return 2
