from hitsz_qy_hummingbird.utils.Optimization.scoring_box import BoxForScoring
from hitsz_qy_hummingbird.configuration import configuration
from hitsz_qy_hummingbird.wing_beat_controller.KSplit_params import ParamsForKSplit
from hitsz_qy_hummingbird.configuration.configuration import GLOBAL_CONFIGURATION
import pandas as pd
import cProfile
import numpy as np

# The First Thing is to find the resonance frequency

# GLOBAL_CONFIGURATION SETUP

GLOBAL_CONFIGURATION.TIMESTEP = 5000
configuration.ParamsForMAV_rl.change_parameters(sleep_time=0)

frequencylist = np.linspace(20,40,11)

Inertia = 3.56e-7
motor_inertia = configuration.ParamsForMaxonSpeed6M_rl.rotor_inertia
kk = configuration.ParamsForMaxonSpeed6M_rl
targetFrequency = np.sqrt(kk.spring_constant/
                          (motor_inertia*kk.gear_ratio*kk.gear_efficiency+Inertia)/4/np.pi/np.pi)
print(kk.spring_constant)
print(motor_inertia)
print(kk.spring_constant/(motor_inertia*kk.gear_ratio*kk.gear_efficiency+Inertia)*4*np.pi*np.pi)
print(targetFrequency)

for fre in frequencylist:
    controllerParams = ParamsForKSplit(nominal_amplitude=15,
                                       frequency=fre,
                                       differential_amplitude=0,
                                       bias_amplitude=0,
                                       split_cycle=0.5)

    TheBox = BoxForScoring(mav_params=configuration.ParamsForMAV_rl,
                           motor_params=configuration.ParamsForMaxonSpeed6M_rl,
                           wing_params=configuration.ParamsForWing_rl,
                           controller=controllerParams.the_according_controller(),
                           periods= 5000)

    # cProfile.run('TheBox.run()')
    TheBox.run()
    TheBox.close()
    print(fre,TheBox.power_loading_scoring())
# data = pd.DataFrame(TheBox.data)
#
# data.to_csv("tem.csv", index=False)

