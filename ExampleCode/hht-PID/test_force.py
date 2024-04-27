'''this file tests the generation of force'''

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
import csv
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import medfilt

#Fz,Tx,Ty,or Tz
test_type = 'Tz'

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

configuration.ParamsForMAV_One.change_parameters(sleep_time=0)

# urdf_creator = URDFCreator(gear_ratio=ng,
#                             aspect_ratio=ar,
#                             taper_ratio=tr,
#                             r22=r22,)
# urdf_name = urdf_creator.write_the_urdf()
temp_urdf = GLOBAL_CONFIGURATION.temporary_urdf_path + f"Ng_{ng}_AR_{ar}_TR_{tr}_R22_{r22}.urdf"

basemav = BaseMavSequential(
                    urdf_name=temp_urdf,
                    mav_params=configuration.ParamsForMAV_One,
                    if_gui=False,
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

data['lift_force'] = []
data['roll_torque'] = []
data['pitch_torque'] = []
data['yaw_torque'] = []

obs, wingobs = mav.reset()

print("base initial obs:\n")
print(obs)
print("wing initial wingbos:\n")
print(wingobs)

gradual = 0
durtime = mav.CTRL_FREQ//2
halfdurtime = durtime//2
if(test_type == 'Fz'):
    delta_voltage_array = np.arange(-5, 5.1, 1)
    testpoints=delta_voltage_array.shape
    mean_Fz = np.zeros(testpoints)
    voltage_array = [v + 10 for v in delta_voltage_array]
if(test_type == 'Tx'):
    differential_voltage_array = np.arange(-2.0, 2.1, 0.1)
    testpoints=differential_voltage_array.shape
    mean_Tx = np.zeros(testpoints)
if(test_type == 'Ty'):
    mean_voltage_array = np.arange(-3.0, 3.1, 0.1)
    testpoints=mean_voltage_array.shape
    mean_Ty = np.zeros(testpoints)
if(test_type == 'Tz'):
    split_cycle_array = np.arange(-0.15, 0.16, 0.01)
    testpoints=split_cycle_array.shape
    mean_Tz = np.zeros(testpoints)

print(testpoints)

for i in range(testpoints[0]):
    cnt=0
    if(test_type == 'Fz'):
        data['lift_force'] = []
        model.d_voltage_amplitude = delta_voltage_array[i]
    if(test_type == 'Tx'):
        data['roll_torque'] = []
        model.differential_voltage = differential_voltage_array[i]
    if(test_type == 'Ty'):
        data['pitch_torque'] = []
        model.mean_voltage = mean_voltage_array[i]
    if(test_type == 'Tz'):
        data['yaw_torque'] = []
        model.split_cycle = split_cycle_array[i]

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
                r_voltage, l_voltage = model.straight_cos()
            else:
                r_voltage, l_voltage = model.predict(obs)
            action = [r_voltage, l_voltage]
            _, _, forceinfo= mav.step(action=action)
        cnt = cnt + 1

        if(test_type == 'Fz'):
            f3= forceinfo[2,:]
            for j in range(f3.shape[0]):
                data['lift_force'].append(-1*f3[j])

        if(test_type == 'Tx'):
            t1= forceinfo[3,:]
            for j in range(t1.shape[0]):
                data['roll_torque'].append(-1*t1[j])

        if(test_type == 'Ty'):
            t2= forceinfo[4,:]
            for j in range(t2.shape[0]):
                data['pitch_torque'].append(-1*t2[j])

        if(test_type == 'Tz'):
            t3= forceinfo[5,:]
            for j in range(t3.shape[0]):
                data['yaw_torque'].append(-1*t3[j])

    if(test_type == 'Fz'):
        mean_Fz[i] = np.array(data["lift_force"][halfdurtime*20:durtime*20]).mean()
        print("Voltage:",model.voltage_amplitude)
        print('mean_Fz is:', mean_Fz[i])

    if(test_type == 'Tx'):
        mean_Tx[i] = np.array(data["roll_torque"][halfdurtime*20:durtime*20]).mean()
        print("differential_voltage:",model.differential_voltage)
        print('mean_Tx is:', mean_Tx[i])

    if(test_type == 'Ty'):
        mean_Ty[i] = np.array(data["pitch_torque"][halfdurtime*20:durtime*20]).mean()
        print("mean_voltage:",model.mean_voltage)
        print('mean_Ty is:', mean_Ty[i])

    if(test_type == 'Tz'):
        mean_Tz[i] = np.array(data["yaw_torque"][halfdurtime*20:durtime*20]).mean()
        print("split_cycle:",model.split_cycle)
        print('mean_Tz is:', mean_Tz[i])


mav.close()

if(test_type == 'Fz'):
    with open('fz_data_3.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Voltage', 'mean_Fz'])
        for voltage, force in zip(voltage_array, mean_Fz):
            writer.writerow([voltage, force])

if(test_type == 'Tx'):
    with open('Tx_data.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['differential_voltage', 'mean_Tx'])
        for dv, tx in zip(differential_voltage_array, mean_Tx):
            writer.writerow([dv, tx])

if(test_type == 'Ty'):
    with open('Ty_data.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['mean_voltage', 'mean_Ty'])
        for mv, ty in zip(mean_voltage_array, mean_Ty):
            writer.writerow([mv, ty])

if(test_type == 'Tz'):
    with open('Tz_data.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['split_cycle', 'mean_Tz'])
        for sc, tz in zip(split_cycle_array, mean_Tz):
            writer.writerow([sc, tz])






