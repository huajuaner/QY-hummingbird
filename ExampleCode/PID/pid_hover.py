'''this file runs the PID control with PIDLinear model when flag_pid=1, and
runs the manually set voltage when flag_pid=0'''

import sys 
sys.path.append('D://graduate//fwmav//simul2024//240325git//QY-hummingbird')
from hitsz_qy_hummingbird.wrapper.trac_wrapped import TracMAV
from hitsz_qy_hummingbird.configuration import configuration
from hitsz_qy_hummingbird.base_FWMAV.motor.motor_params import ParamsForBaseMotor
from hitsz_qy_hummingbird.base_FWMAV.wings.wing_params import ParamsForBaseWing
from hitsz_qy_hummingbird.pid_controller.pidlinear import PIDLinear
from hitsz_qy_hummingbird.configuration.configuration import GLOBAL_CONFIGURATION
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

GLOBAL_CONFIGURATION.logger_init()

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

mav_params = configuration.ParamsForMAV_One.change_parameters(sleep_time=0.001)

mav = TracMAV(mav_params=configuration.ParamsForMAV_One,
                 motor_params=motor_params,
                 wing_params=wing_params)

flag_pid = 0
model = PIDLinear()

model.pos_target_x = 0
model.pos_target_y = 0
model.pos_target_z = 1
model.ang_ef_target_yaw = 0

#print("aaaaaaaaaaaaaaaaa_body_unique_id is        \n" + str(mav.mav.body_unique_id))

data = {}
data['right_stroke_amp'] = []
data['left_stroke_amp'] = []
data['right_stroke_vel'] = []
data['left_stroke_vel'] = []
data['r_u']=[]
#data['r_t']=[]
data['l_u']=[]
#data['l_t']=[]

obs, wingobs = mav.reset()

print("base initial obs:\n")
print(obs)
print("wing initial wingbos:\n")
print(wingobs)

cnt = 0
gradual = 230
while cnt < int(3*mav.CTRL_FREQ) :

    if(cnt<gradual):
        if flag_pid == 0:
            r_voltage, l_voltage = model.straight_cos() 
            r_voltage = r_voltage*cnt/gradual
            l_voltage = l_voltage*cnt/gradual
        else:
            r_voltage, l_voltage = model.predict(obs)
        action = [r_voltage, l_voltage]
        obs, wingobs = mav.step(action=action)
        
    if(cnt == gradual):
        print("gradual end obs:\n")
        print(obs)

    if(cnt>gradual-1):
        if flag_pid == 0:
            r_voltage, l_voltage = model.straight_cos() 
        else:
            r_voltage, l_voltage = model.predict(obs)
        action = [r_voltage, l_voltage]
        obs, wingobs = mav.step(action=action)

    data['right_stroke_amp'] .append(wingobs[0])
    data['left_stroke_amp'] .append(wingobs[1])
    data['right_stroke_vel'] .append(wingobs[2])
    data['left_stroke_vel'] .append(wingobs[3])
    data['r_u'].append(r_voltage)
    #data['r_t'].append(r_t)
    data['l_u'].append(l_voltage)
    #data['l_t'].append(l_t)

    cnt = cnt + 1


print("base end obs:\n")
print(obs)
print("wing end wingbos:\n")
print(wingobs)
       
mav.close()
data = pd.DataFrame(
    data
)
data.to_csv("tem.csv",index = False)

# plt.plot(data.index, data['right_stroke_amp'], label='Right Stroke Amp')
# plt.plot(data.index, data['left_stroke_amp'], label='Left Stroke Amp')
# plt.xlabel('Index')
# plt.ylabel('Amplitude')
# plt.title('Stroke Amplitude')
# plt.legend()
# plt.show()

fig, ax1 = plt.subplots(facecolor='lightblue')

color1 = 'tab:red'
color2 = 'tab:blue'
color3 = 'tab:green'
color4 = 'tab:orange'
ax1.set_xlabel('time (step)')
ax1.set_ylabel('degree(°)', color=color2)
#ax1.plot(data.index,  180*data['right_stroke_amp']/np.pi, color=color1, linewidth=4.0)
ax1.plot(data.index,  180*data['left_stroke_amp']/np.pi, color=color2, linewidth=4.0)
ax1.tick_params(axis='y', labelcolor=color2)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

ax2.set_ylabel('voltage(V)', color=color4)  # we already handled the x-label with ax1
#ax2.plot(data.index, data['r_u'], color=color3)
ax2.plot(data.index, data['l_u'], color=color4)
ax2.tick_params(axis='y', labelcolor=color4)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()