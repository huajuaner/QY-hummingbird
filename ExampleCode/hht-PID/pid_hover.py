'''This file runs the hht-PID control with PIDLinear model when flag_pid=1, and
runs the manually set voltage when flag_pid=0.
Record the force info during per control step.'''

import sys

sys.path.append('/home/hht/simul0422/240414/QY-hummingbird')

from hitsz_qy_hummingbird.wrapper.wrapped_mav_for_PID import WrappedMAVpid
from hitsz_qy_hummingbird.base_FWMAV.MAV.base_MAV_sequential import BaseMavSequential
from hitsz_qy_hummingbird.configuration import configuration
from hitsz_qy_hummingbird.base_FWMAV.motor.motor_params import ParamsForBaseMotor
from hitsz_qy_hummingbird.base_FWMAV.wings.wing_params import ParamsForBaseWing
from hitsz_qy_hummingbird.pid_controller.pidlinear import PIDLinear
from hitsz_qy_hummingbird.configuration.configuration import GLOBAL_CONFIGURATION
from hitsz_qy_hummingbird.utils.create_urdf_2 import URDFCreator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import medfilt

# GLOBAL_CONFIGURATION.logger_init()

ng =10
ar =5.2
tr =0.8
r22 =4e-6
motor_params = configuration.ParamsForMaxonSpeed6M
motor_params.change_parameters(spring_wire_diameter=0.7,
                               spring_number_of_coils=6,
                               spring_outer_diameter=3.5,
                               gear_efficiency=0.8,
                               gear_ratio=ng)

wing_params = ParamsForBaseWing(aspect_ratio=ar,
                                taper_ratio=tr,
                                r22=r22,
                                camber_angle=16 / 180 * np.pi,
                                resolution=500)

configuration.ParamsForMAV_One.change_parameters(sleep_time=0.0001)

# urdf_creator = URDFCreator(gear_ratio=ng,
#                             aspect_ratio=ar,
#                             taper_ratio=tr,
#                             r22=r22,)
# urdf_name = urdf_creator.write_the_urdf()
temp_urdf = GLOBAL_CONFIGURATION.temporary_urdf_path + f"Ng_{ng}_AR_{ar}_TR_{tr}_R22_{r22}.urdf"

basemav = BaseMavSequential(
                    urdf_name=temp_urdf,
                    mav_params=configuration.ParamsForMAV_One,
                    if_gui=True,
                    if_fixed=True,
)

mav = WrappedMAVpid(basemav,
                   motor_params=motor_params,
                   wing_params=wing_params)

flag_pid = 0
model = PIDLinear()

model.pos_target_x = 0
model.pos_target_y = 0
model.pos_target_z = 1
model.ang_ef_target_yaw = 0

# print("aaaaaaaaaaaaaaaaa_body_unique_id is        \n" + str(mav.mav.body_unique_id))

data = {}
data['right_stroke_amp'] = []
data['left_stroke_amp'] = []
data['right_rotate_amp'] = []
data['left_rotate_amp'] = []
# data['right_stroke_vel'] = []
# data['left_stroke_vel'] = []
data['r_u'] = []
data['l_u'] = []
data['lift_force'] = []
data['roll_torque'] = []
data['pitch_torque'] = []
data['yaw_torque'] = []

obs, wingobs = mav.reset()

f01, f02, f03, t01, t02, t03=mav.get_force_info()
print("力",f01, f02, f03, t01, t02, t03)


print("base initial obs:\n")
print(obs)
print("wing initial wingbos:\n")
print(wingobs)

cnt = 0
wingbeat = 6
gradual = 0
durtime = mav.CTRL_FREQ//2
while cnt < int (durtime):

    # if (cnt < gradual):
    #     if flag_pid == 0:
    #         r_voltage, l_voltage = model.straight_cos()
    #         r_voltage = r_voltage * cnt / gradual
    #         l_voltage = l_voltage * cnt / gradual
    #     else:
    #         r_voltage, l_voltage = model.predict(obs)
    #     action = [r_voltage, l_voltage]
    #     obs, wingobs = mav.step(action=action)

    # if (cnt == gradual):
    #     print("gradual end obs:\n")
    #     print(obs)

    if (cnt > gradual - 1):
        if flag_pid == 0:
            r_voltage = 0
            _,l_voltage = model.straight_cos()
        else:
            r_voltage, l_voltage = model.predict(obs)
        action = [r_voltage, l_voltage]
        obs, wingobs= mav.step_no_forceinfo(action=action)

    f1, f2, f3, t1, t2, t3=mav.get_force_info()

    data['right_stroke_amp'].append(wingobs[0])
    data['left_stroke_amp'].append(wingobs[1])
    data['right_rotate_amp'].append(wingobs[2])
    data['left_rotate_amp'].append(wingobs[3])
    data['r_u'].append(r_voltage)
    # data['r_t'].append(r_t)
    data['l_u'].append(l_voltage)
    # data['l_t'].append(l_t)
    data['lift_force'].append(-1*f3)
    data['roll_torque'].append(t1)
    data['pitch_torque'].append(t2)
    data['yaw_torque'].append(t3)

    cnt = cnt + 1

print("base end obs:\n")
print(obs)
print("wing end wingbos:\n")
print(wingobs)

mav.close()
data = pd.DataFrame(
    data
)
data.to_csv("tem.csv", index=False)

smoothed_lift_force = medfilt(data['lift_force'], kernel_size=3)
smoothed_roll_torque = medfilt(data['roll_torque'], kernel_size=5)
smoothed_pitch_torque = medfilt(data['pitch_torque'], kernel_size=3)
smoothed_yaw_torque = medfilt(data['yaw_torque'], kernel_size=3)

# plt.plot(data.index, data['right_stroke_amp'], label='Right Stroke Amp')
# plt.plot(data.index, data['left_stroke_amp'], label='Left Stroke Amp')
# plt.xlabel('Index')
# plt.ylabel('Amplitude')
# plt.title('Stroke Amplitude')
# plt.legend()
# plt.show()

# fig, ax1 = plt.subplots(facecolor='lightblue')

color1 = 'tab:red'
color2 = 'tab:blue'
color3 = 'tab:green'
color4 = 'tab:orange'
color_deep_red = (0.7, 0.0, 0.0)
color_deep_blue = (0.0, 0.0, 0.7)
# ax1.set_xlabel('time (control step)')
# ax1.set_ylabel('degree(°)', color=color2)
# # ax1.plot(data.index,  180*data['right_stroke_amp']/np.pi, color=color1, linewidth=4.0)
# ax1.plot(data.index, 180 * data['left_stroke_amp'] / np.pi, color=color2, linewidth=4.0)
# ax1.tick_params(axis='y', labelcolor=color2)

# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

# ax2.set_ylabel('voltage(V)', color=color4)  # we already handled the x-label with ax1
# # ax2.plot(data.index, data['r_u'], color=color3)
# ax2.plot(data.index, data['l_u'], color=color4)
# ax2.tick_params(axis='y', labelcolor=color4)

# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.show()



fig2, ax3 = plt.subplots(3, 2, figsize=(10, 6))

fig2.subplots_adjust(left=0.067, right=0.948,top=0.962,bottom=0.067, wspace=0.32,hspace=0.54)
alp_size=16
tick_size = 16
#### Column ################################################
col = 0

####  ###################################################
row = 0

ax3[row, col].plot(data.index/1200, data['l_u'], color='tab:red' , linewidth=1)
ax3[row, col].set_xlabel('time(s)',loc='right', fontsize=alp_size)
ax3[row, col].set_ylabel('voltage(V)', color='tab:red', fontsize=alp_size) 
ax3[row, col].tick_params(axis='x', labelsize=tick_size )
ax3[row, col].tick_params(axis='y', labelcolor='tab:red', labelsize=tick_size)
ax3[row, col].set_yticks([-30,-20,-10,0,10,20,30])
# ax3[row, col].text(durtime/2, -5, '(' + chr(97) + ')', fontsize=12, ha='left',va='baseline')

ax4 = ax3[row, col].twinx()
ax4.plot(data.index/1200, 180 * data['left_stroke_amp'] / np.pi, color='tab:orange', linewidth=1)
ax4.set_ylabel('stroke angle(°)', color='tab:orange', fontsize=alp_size)
ax4.set_yticks([-75,-50,-25,0,25,50,75])
ax4.tick_params(axis='y', labelcolor='tab:orange', labelsize=tick_size)

row = 1

ax3[row, col].plot(data.index/1200, smoothed_lift_force)
ax3[row, col].set_xlabel('time(s)',loc='right', fontsize=alp_size)
ax3[row, col].set_ylabel('$F_z$ (N)', fontsize=alp_size)
ax3[row, col].set_yticks([-0.2,-0.1, 0, 0.1, 0.2, 0.3, 0.4])
ax3[row, col].tick_params(axis='x', labelsize=tick_size )
ax3[row, col].tick_params(axis='y', labelsize=tick_size )

row = 2

ax3[row, col].plot(data.index/1200, 1e3*smoothed_pitch_torque)
ax3[row, col].set_xlabel('time(s)',loc='right', fontsize=alp_size)
ax3[row, col].set_ylabel('pitch torque (N$\\cdot$mm)', fontsize=alp_size)
ax3[row, col].tick_params(axis='x', labelsize=tick_size )
ax3[row, col].tick_params(axis='y', labelsize=tick_size )


#### Column ################################################
col = 1

#### XYZ ###################################################
row = 0

# ax3[row, col].plot(data.index/1200, data['r_u'], color='tab:red' , linewidth=1)
# ax3[row, col].set_xlabel('time(s)',loc='right', fontsize=alp_size)
# ax3[row, col].set_ylabel('right voltage(V)', color='tab:red' , fontsize=alp_size) 
# ax3[row, col].tick_params(axis='x', labelsize=tick_size )
# ax3[row, col].tick_params(axis='y', labelcolor='tab:red', labelsize=tick_size)
# ax3[row, col].set_yticks([-30,-20,-10,0,10,20,30])

# ax5 = ax3[row, col].twinx()
# ax5.plot(data.index/1200, 180 * data['right_stroke_amp'] / np.pi, color='tab:orange', linewidth=1)
# ax5.set_ylabel('right stroke angle(°)', color='tab:orange', fontsize=alp_size)
# ax5.set_yticks([-75,-50,-25,0,25,50,75])
# ax5.tick_params(axis='y', labelcolor='tab:orange', labelsize=tick_size)
ax3[row, col].plot(data.index/1200, data['l_u'], color='tab:red' , linewidth=1)
ax3[row, col].set_xlabel('time(s)',loc='right', fontsize=alp_size)
ax3[row, col].set_ylabel('voltage(V)', color='tab:red' , fontsize=alp_size) 
ax3[row, col].tick_params(axis='x', labelsize=tick_size )
ax3[row, col].tick_params(axis='y', labelcolor='tab:red', labelsize=tick_size)
ax3[row, col].set_yticks([-30,-20,-10,0,10,20,30])

ax5 = ax3[row, col].twinx()
ax5.plot(data.index/1200, 180 * data['left_rotate_amp'] / np.pi, color='tab:green', linewidth=1)
ax5.set_ylabel('rotation angle(°)', color='tab:green', fontsize=alp_size)
ax5.set_yticks([-75,-50,-25,0,25,50,75])
ax5.tick_params(axis='y', labelcolor='tab:green', labelsize=tick_size)

row = 1

ax3[row, col].plot(data.index/1200, 1e3*smoothed_roll_torque)
ax3[row, col].set_xlabel('time(s)',loc='right', fontsize=alp_size)
ax3[row, col].set_ylabel('roll torque (N$\\cdot$mm)', fontsize=alp_size)
ax3[row, col].set_yticks([-0.1,0,0.1])
ax3[row, col].tick_params(axis='x', labelsize=tick_size )
ax3[row, col].tick_params(axis='y', labelsize=tick_size )
# # 调整科学计数法的字体大小
# ax3[row, col].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
ax3[row, col].yaxis.offsetText.set_fontsize(tick_size)

row = 2

ax3[row, col].plot(data.index/1200, 1e3*smoothed_yaw_torque)
ax3[row, col].set_xlabel('time(s)',loc='right', fontsize=alp_size)
ax3[row, col].set_ylabel('yaw torque (N$\\cdot$mm)', fontsize=alp_size)
ax3[row, col].set_yticks([-0.1,0,0.1])
ax3[row, col].tick_params(axis='x', labelsize=tick_size )
ax3[row, col].tick_params(axis='y', labelsize=tick_size )
# 调整科学计数法的字体大小
# ax3[row, col].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
ax3[row, col].yaxis.offsetText.set_fontsize(tick_size)

plt.show()
