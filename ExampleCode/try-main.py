import sys 
sys.path.append('D://graduate//fwmav//simul2024//240328git//QY-hummingbird')
from hitsz_qy_hummingbird.utils.Optimization.scoring_box import BoxForScoring
from hitsz_qy_hummingbird.configuration import configuration
from hitsz_qy_hummingbird.wing_beat_controller.KSplit_params import ParamsForKSplit
from hitsz_qy_hummingbird.configuration.configuration import GLOBAL_CONFIGURATION
from hitsz_qy_hummingbird.base_FWMAV.wings.wing_params import ParamsForBaseWing
import pandas as pd
import cProfile
import numpy as np

# The First Thing is to find the resonance frequency

# GLOBAL_CONFIGURATION SETUP

GLOBAL_CONFIGURATION.TIMESTEP = 5000
configuration.ParamsForMAV_rl.change_parameters(sleep_time=0)

#frequency_list = np.arange(12, 18, 0.02)
frequency_list = np.arange(25, 35, 1)

Inertia = 4.16e-7
motor_inertia = configuration.ParamsForMaxonSpeed6M_rl.rotor_inertia
kk = configuration.ParamsForMaxonSpeed6M_rl
targetFrequency = np.sqrt(kk.spring_constant/
                          (motor_inertia*kk.gear_ratio*kk.gear_efficiency+Inertia)/4/np.pi/np.pi)
print(kk.spring_constant)
print(motor_inertia)
print(kk.spring_constant/(motor_inertia*kk.gear_ratio*kk.gear_efficiency+Inertia)*4*np.pi*np.pi)
print(targetFrequency)


ng =10
ar =5.2
tr =0.85
r22 =4e-6

tempourdf = GLOBAL_CONFIGURATION.temporary_urdf_path + f"Ng_{ng}_AR_{ar}_TR_{tr}_R22_{r22}.urdf"

for fre in frequency_list:
    controllerParams = ParamsForKSplit(nominal_amplitude=15,
                                       frequency=fre,
                                       differential_amplitude=0,
                                       bias_amplitude=0,
                                       split_cycle=0.5)

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
    
    TheBox = BoxForScoring(urdf_name = tempourdf,
                           mav_params=configuration.ParamsForMAV_One,
                           motor_params=motor_params,
                           wing_params=wing_params,
                           controller=controllerParams.the_according_controller(),
                           periods= 5000)

    # cProfile.run('TheBox.run()')
    TheBox.run()
    TheBox.close()
    a,b=TheBox.power_loading_scoring()
    if  a>GLOBAL_CONFIGURATION.MAXlift:
        GLOBAL_CONFIGURATION.MAXlift=a
        GLOBAL_CONFIGURATION.Bestf = fre
        print('newbee')
    print(fre,a,b)
print(GLOBAL_CONFIGURATION.Bestf,GLOBAL_CONFIGURATION.MAXlift)
# data = pd.DataFrame(TheBox.data)
#
# data.to_csv("tem.csv", index=False)

