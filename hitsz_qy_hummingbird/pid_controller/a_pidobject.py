import numpy as np
from hitsz_qy_hummingbird.configuration.configuration import GLOBAL_CONFIGURATION


class PidObj:

    def __init__(self, integ_max, Kp, Ki, Kd):
        self.old_error = 0
        self.integ = 0
        self.integ_max = integ_max
        self.timestep = 1 / GLOBAL_CONFIGURATION.TIMESTEP
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

    def pid_update(self, error):
        integ = np.clip(self.integ + error * self.timestep, -self.integ_max, self.integ_max)
        deriv = (error - self.old_error) / self.timestep

        self.old_error = error
        self.integ = integ

        return self.Kp * error + self.Ki * integ + self.Kd * deriv
