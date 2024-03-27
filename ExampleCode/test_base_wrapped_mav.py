'''
This file is used as the standard test file of the base_wrapped_mav
'''
import sys 
sys.path.append('D://graduate//fwmav//simul2024//240325git//QY-hummingbird')
import time
from hitsz_qy_hummingbird.wrapper.base_wrapped_mav import BaseWrappedMAV
from hitsz_qy_hummingbird.configuration import configuration
from hitsz_qy_hummingbird.base_FWMAV.motor.motor_params import ParamsForBaseMotor
from hitsz_qy_hummingbird.base_FWMAV.wings.wing_params import ParamsForBaseWing
from hitsz_qy_hummingbird.wing_beat_controller.synchronous_controller_Ksquare import SynchronousControllerKsquare
from hitsz_qy_hummingbird.wing_beat_controller.DIY_controller_for_four_bar import DIY_controller_for_Four_Bar
from hitsz_qy_hummingbird.configuration.configuration import GLOBAL_CONFIGURATION
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

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

mav_params = configuration.ParamsForMAV_One.change_parameters(sleep_time=0.000001)

mav = BaseWrappedMAV(mav_params=configuration.ParamsForMAV_One,
                     motor_params=motor_params,
                     wing_params=wing_params,
                     if_gui=True,
                     if_fixed=False)

# controller = SynchronousControllerKsquare(nominal_amplitude=np.pi / 3,
#                                           frequency=30)

controller = DIY_controller_for_Four_Bar(nominal_amplitude=np.pi / 3,
                                         frequency=20,
                                         differential_frequency=0,
                                         right_split_cycle=0.30,
                                         left_split_cycle=0.70,
                                         bias=0.3)
cnt = 0
time_lis = []
target_right_stroke_amp_lis = []
target_right_stroke_vel_lis = []
start_time = time.time()
period = 1000

while cnt < 1000:
    cnt = cnt + 1
    if cnt % 50 == 0:
        elapsed_time = time.time() - start_time
        sys.stdout.write('\r')
        sys.stdout.write("[%-20s] %d%% Elapsed Time: %ds" % ('=' * int(cnt / 50), int(cnt / 10), elapsed_time))
        sys.stdout.flush()
    (target_right_stroke_amp,
     target_right_stroke_vel,
     target_left_stroke_amp,
     target_left_stroke_vel) = controller.step()
    target_right_stroke_amp_lis.append(target_right_stroke_amp)
    target_right_stroke_vel_lis.append(target_right_stroke_vel)
    action = [target_right_stroke_amp,
              target_right_stroke_vel,
              None,
              target_left_stroke_amp,
              target_left_stroke_vel,
              None]
    states = mav.step(action=action)
    time_lis.append(GLOBAL_CONFIGURATION.TIME)
    GLOBAL_CONFIGURATION.step()

mav.close()
data = pd.DataFrame(
    {
        # 'x_force': mav.mav.data['x_force'],
        # 'y_force': mav.mav.data['y_force'],
        # 'z_force': mav.mav.data['z_force'],
        # 'x_torque': mav.mav.data['x_torque'],
        # 'y_torque': mav.mav.data['y_torque'],
        # 'z_torque': mav.mav.data['z_torque'],
        'right_stroke_amp': mav.mav.data['right_stroke_amp_lis'],
        'right_stroke_vel': mav.mav.data['right_stroke_vel_lis'],
        'left_stroke_amp': mav.mav.data['left_stroke_amp_lis'],
        'left_stroke_vel': mav.mav.data['left_stroke_vel_lis'],
        'target_right_stroke_amp': controller.data['target_right_stroke_amp_lis'],
        'target_right_stroke_vel': controller.data['target_right_stroke_vel_lis'],
        'target_left_stroke_amp': controller.data['target_left_stroke_amp_lis'],
        'target_left_stroke_vel': controller.data['target_left_stroke_vel_lis'],
        'aa': controller.data['seeseelist'],
        # 'right_motor_torque': mav.right_motor.data['torque_motor'],
        # 'right_output_torque_without_SpringandDamping':mav.mav.data['right_torque_without_SpringandDamping'],
        # 'right_motor_current': mav.right_motor.data['current'],
        # 'right_motor_voltage': mav.right_motor.data['voltage'],
        # 'right_motor_output_power': mav.right_motor.data['output_power'],
        # 'right_motor_efficiency_1': mav.right_motor.data['efficiency1'],
        # 'right_motor_efficiency_2': mav.right_motor.data['efficiency2'],
        # 'right_motor_efficiency_3': mav.right_motor.data['efficiency3'],
        # 'target_right_stroke_amp': target_right_stroke_amp_lis,
        # 'target_right_stroke_vel': target_right_stroke_vel_lis,
        'time': time_lis
    }
)

data.to_csv("tem.csv", index=False)

# fig, ax1 = plt.subplots(figsize=(8, 6))
# ax1.plot(data.index[200:],
#          data['x_force'][200:],
#          label='x_force')
# ax1.plot(data.index[200:],
#          data['y_force'][200:],
#          label='y_force')
# ax1.plot(data.index[200:],
#          data['z_force'][200:],
#          label='z_force')
# ax1.legend()
# ax2 = ax1.twinx()
# ax2.plot(data.index[200:],
#          data['right_stroke_amp'][200:])
# plt.xlabel('Index')
# # plt.ylabel('force(N)')
# plt.title('fxxxk')
# # plt.legend()
# plt.show()
