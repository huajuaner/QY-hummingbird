from hitsz_qy_hummingbird.wrapper.clamped_wrapped import ClampedMAV
from hitsz_qy_hummingbird.configuration import configuration
from hitsz_qy_hummingbird.base_FWMAV.motor.motor_params import ParamsForBaseMotor
from hitsz_qy_hummingbird.base_FWMAV.wings.wing_params import ParamsForBaseWing
from hitsz_qy_hummingbird.wing_beat_controller.synchronous_controller import WingBeatProfile
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

mav = ClampedMAV(mav_params=configuration.ParamsForMAV_One,
                 motor_params=motor_params,
                 wing_params=wing_params)

controller = WingBeatProfile(nominal_amplitude=np.pi / 3,
                             frequency=30)

data = {}
data['right_stroke_amp'] = []
data['left_stroke_amp'] = []
data['right_stroke_vel'] = []
data['left_stroke_vel'] = []

cnt = 0
while cnt < 1000:
    cnt = cnt + 1
    (right_stroke_amp, right_stroke_vel, _, left_stroke_amp, left_stroke_vel, _) = controller.step()
    data['right_stroke_amp'].append(right_stroke_amp)
    data['left_stroke_amp'].append(left_stroke_amp)
    data['right_stroke_vel'] . append( right_stroke_vel)
    data['left_stroke_vel'] . append( left_stroke_vel)
    action = [right_stroke_amp, None, None, left_stroke_amp, None, None]
    mav.step(action=action)

mav.close()
data = pd.DataFrame(
    data
)
data.to_csv("tem.csv",index = False)
plt.plot(data.index, data['right_stroke_amp'], label='Right Stroke Amp')
plt.plot(data.index, data['left_stroke_amp'], label='Left Stroke Amp')
plt.xlabel('Index')
plt.ylabel('Amplitude')
plt.title('Stroke Amplitude')
plt.legend()
plt.show()