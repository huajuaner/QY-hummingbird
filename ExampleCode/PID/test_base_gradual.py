'''
Apply SynchronousControllerKsquare contorller.
Given gradually increasing amplitudes.
'''
import sys 
sys.path.append('D://graduate//fwmav//simul2024//240325git//QY-hummingbird')
from hitsz_qy_hummingbird.wrapper.base_wrapped_mav import BaseWrappedMAV
from hitsz_qy_hummingbird.configuration import configuration
from hitsz_qy_hummingbird.base_FWMAV.motor.motor_params import ParamsForBaseMotor
from hitsz_qy_hummingbird.base_FWMAV.wings.wing_params import ParamsForBaseWing
from hitsz_qy_hummingbird.wing_beat_controller.synchronous_controller_Ksquare import SynchronousControllerKsquare
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hitsz_qy_hummingbird.configuration.configuration import GLOBAL_CONFIGURATION


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

mav = BaseWrappedMAV(
                    if_gui=True,
                    if_fixed=False,
                    mav_params=configuration.ParamsForMAV_One,
                    motor_params=motor_params,
                    wing_params=wing_params)

print("a_body_unique_id is        \n" + str(mav.mav.body_unique_id))

#np.pi *5 / 12
controller = SynchronousControllerKsquare(nominal_amplitude=np.pi /3,
                             frequency=34)

data = {}
# data['x'] = []
# data['y'] = []
# data['z'] = []
# data['roll'] = []
# data['pitch'] = []
# data['yaw'] = []
data['right_stroke_amp'] = []
data['left_stroke_amp'] = []
data['right_stroke_vel'] = []
data['left_stroke_vel'] = []
data['r_u']=[]
data['r_t']=[]
data['l_u']=[]
data['l_t']=[]

cnt = 0
gradual = 1050
while cnt < 10000:
    cnt = cnt + 1
    if(cnt != gradual):
        (right_stroke_amp, right_stroke_vel, left_stroke_amp, left_stroke_vel) = controller.step()

    if(cnt<gradual):
        action = [right_stroke_amp*cnt/gradual, None, None,left_stroke_amp*cnt/gradual, None, None]
        _,_,_,r_t,_,r_u,_,_,_,l_t,_,l_u,_,_,_,_,_,_ = mav.step(action=action)
        
    if(cnt>gradual):
        action = [right_stroke_amp, None, None,left_stroke_amp, None, None]
        _,_,_,r_t,_,r_u,_,_,_,l_t,_,l_u,_,_,_,_,_,_ = mav.step(action=action)

    if(cnt != gradual):
        # data['x'].append(pos[0])
        # data['y'].append(pos[1])
        # data['z'].append(pos[2])
        # data['roll'].append(orn[0]*180/np.pi)
        # data['pitch'].append(orn[1]*180/np.pi)
        # data['yaw'].append(orn[2]*180/np.pi)
        data['right_stroke_amp'].append(right_stroke_amp)
        data['left_stroke_amp'].append(left_stroke_amp)
        data['right_stroke_vel'].append( right_stroke_vel)
        data['left_stroke_vel'].append( left_stroke_vel)
        data['r_u'].append(r_u)
        data['r_t'].append(r_t)
        data['l_u'].append(l_u)
        data['l_t'].append(l_t)
    
    GLOBAL_CONFIGURATION.step()

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


#这里的data.index是上面的cnt时间步
fig, ax1 = plt.subplots(facecolor='lightblue')

color1 = 'tab:red'
color2 = 'tab:blue'
color3 = 'tab:orange'
color4 = 'tab:green'
ax1.set_xlabel('time (step)')
ax1.set_ylabel('degree(°)', color=color1)
ax1.plot(data.index,  180*data['right_stroke_amp']/np.pi, color=color1)
ax1.plot(data.index,  180*data['left_stroke_amp']/np.pi, color=color2)
ax1.tick_params(axis='y', labelcolor=color1)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

ax2.set_ylabel('voltage(V)', color=color3)  # we already handled the x-label with ax1
ax2.plot(data.index, data['r_u'], color=color3, linewidth=7.0)
ax2.plot(data.index, data['l_u'], color=color4, linewidth=7.0)
ax2.tick_params(axis='y', labelcolor=color3)

fig.tight_layout()  # otherwise the right y-label is slightly clipps
plt.show()

# fig2, ax3 = plt.subplots(3, 2)

# #### Column ################################################
# col = 0

# #### XYZ ###################################################
# row = 0

# ax3[row, col].plot(data.index, data['x'])
# ax3[row, col].set_xlabel('time')
# ax3[row, col].set_ylabel('x (m)')

# row = 1

# ax3[row, col].plot(data.index, data['y'])
# ax3[row, col].set_xlabel('time')
# ax3[row, col].set_ylabel('y (m)')

# row = 2

# ax3[row, col].plot(data.index, data['z'])
# ax3[row, col].set_xlabel('time')
# ax3[row, col].set_ylabel('z (m)')

# #### Column ################################################
# col = 1

# #### XYZ ###################################################
# row = 0

# ax3[row, col].plot(data.index, data['roll'])
# ax3[row, col].set_xlabel('time')
# ax3[row, col].set_ylabel('roll (deg)')

# row = 1

# ax3[row, col].plot(data.index, data['pitch'])
# ax3[row, col].set_xlabel('time')
# ax3[row, col].set_ylabel('pitch (deg)')

# row = 2

# ax3[row, col].plot(data.index, data['yaw'])
# ax3[row, col].set_xlabel('time')
# ax3[row, col].set_ylabel('yaw (deg)')

# plt.show()