# import sys 
# sys.path.append('D:/graduate/fwmav/Simul2023/230830/QY-hummingbird')

from hitsz_qy_hummingbird.wrapper.wrapped_mav_for_design import WrappedMAVDesign
from hitsz_qy_hummingbird.configuration import configuration
from hitsz_qy_hummingbird.base_FWMAV.motor.motor_params import ParamsForBaseMotor
from hitsz_qy_hummingbird.base_FWMAV.wings.wing_params import ParamsForBaseWing
from hitsz_qy_hummingbird.wing_beat_controller.synchronous_controller_Ksquare import SynchronousControllerKsquare
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

motor_params = configuration.ParamsForMaxonSpeed6M
motor_params.change_parameters(spring_wire_diameter=0.5,
                               spring_number_of_coils=6,
                               spring_outer_diameter=3.5,
                               gear_efficiency=0.8,
                               gear_ratio=10)

wing_params = ParamsForBaseWing(aspect_ratio=9.3,
                                taper_ratio=0.66,
                                r22=4E-6,
                                camber_angle=16 / 180 * np.pi,
                                resolution=500)
mav_params = configuration.ParamsForMAV_One.change_parameters(sleep_time=0.1)

mav = WrappedMAVDesign(mav_params=configuration.ParamsForMAV_One,
                       motor_params=motor_params,
                       wing_params=wing_params)

controller = SynchronousControllerKsquare(nominal_amplitude=np.pi / 3,
                                          frequency=50)

data = {}
data['right_stroke_amp'] = []
data['left_stroke_amp'] = []
data['right_stroke_vel'] = []
data['left_stroke_vel'] = []

cnt = 0
while cnt < 3000:
    cnt = cnt + 1
    (right_stroke_amp, right_stroke_vel, _, left_stroke_amp, left_stroke_vel, _) = controller.step()
    data['right_stroke_amp'].append(right_stroke_amp)
    data['left_stroke_amp'].append(left_stroke_amp)
    data['right_stroke_vel'].append(right_stroke_vel)
    data['left_stroke_vel'].append(left_stroke_vel)
    action = [right_stroke_amp, None, None, None, None, None]
    mav.step(action=action)

mav.close()
