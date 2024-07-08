from hitsz_qy_hummingbird.configuration.global_configuration import GlobalConfigWrapper
from hitsz_qy_hummingbird.base_FWMAV.MAV.MAV_params import ParamsForBaseMAV
from hitsz_qy_hummingbird.base_FWMAV.motor.motor_params import ParamsForBaseMotor
from hitsz_qy_hummingbird.base_FWMAV.wings.wing_params import ParamsForBaseWing
import numpy as np
from numpy import pi

"""
This file stores most of the configuration used while the simulation
"""

GLOBAL_CONFIGURATION = GlobalConfigWrapper()

"""
GLOBAL_CONFIGURATION stores the filepath and the logger.
"""

ParamsForMaxonSpeed6M = ParamsForBaseMotor(terminal_r=12.4,
                                           terminal_l=0.118,
                                           torque_constant=1.75e-3,
                                           speed_constant=5460,
                                           rotor_inertia=7.03e-10,
                                           rotor_damping_coeff=1e-7,
                                           wing_damping_coeff=1e-7,
                                           nominal_current=0.97,
                                           nominal_voltage=12)

# ParamsForMAV_One = ParamsForBaseMAV(rotate_axis=0,
#                                     stroke_axis=2,
#                                     the_left_axis=1,
#                                     gravity=np.array([0, 0, -9.8]),
#                                     sleep_time=0.0001,
#                                     max_angle_of_rotate=pi / 4,
#                                     max_angle_of_stroke=pi / 2,
#                                     max_force=1000,
#                                     max_joint_velocity=1000,
#                                     initial_xyz=np.array([0, 0, 10]),
#                                     initial_rpy=np.array([0, 0, pi/2]),
#                                     right_stroke_joint=0,
#                                     right_rotate_joint=1,
#                                     left_stroke_joint=2,
#                                     left_rotate_joint=3,
#                                     right_rod_link=0,
#                                     right_wing_link=1,
#                                     left_rod_link=2,
#                                     left_wing_link=3,
#                                     position_gain = 0,
#                                     velocity_gain = 0)

# if use 2024new urdf
ParamsForMAV_One = ParamsForBaseMAV(rotate_axis=1,
                                    stroke_axis=2,
                                    the_left_axis=0,
                                    gravity=np.array([0, 0, -9.8]),
                                    sleep_time=0.01,
                                    max_angle_of_rotate=pi / 4,
                                    max_angle_of_stroke=pi / 2,
                                    max_force=1000,
                                    max_joint_velocity=100000,
                                    initial_xyz=np.array([1, 0, 0.5]),
                                    initial_rpy=np.array([0, 0, 0]),
                                    right_stroke_joint=0,
                                    right_rotate_joint=1,
                                    left_stroke_joint=2,
                                    left_rotate_joint=3,
                                    right_rod_link=0,
                                    right_wing_link=1,
                                    left_rod_link=2,
                                    left_wing_link=3,
                                    position_gain=0.4,
                                    velocity_gain=0.1)

# params for hht-RL
ParamsForMaxonSpeed6M_rl = ParamsForBaseMotor(terminal_r=12.4,
                                              terminal_l=0.118,
                                              torque_constant=1.75e-3,
                                              speed_constant=5460,
                                              rotor_inertia=7.03e-10,
                                              rotor_damping_coeff=1e-7,
                                              wing_damping_coeff=1e-7,
                                              nominal_current=0.97,
                                              nominal_voltage=12,
                                              spring_wire_diameter=0.7,
                                              spring_number_of_coils=6,
                                              spring_outer_diameter=3.5,
                                              gear_efficiency=0.8,
                                              gear_ratio=10)

ParamsForWing_rl = ParamsForBaseWing(aspect_ratio=5.2,
                                     taper_ratio=0.8,
                                     r22=4E-6,
                                     camber_angle=16 / 180 * np.pi,
                                     resolution=500)

ParamsForMAV_rl = ParamsForBaseMAV(rotate_axis=1,
                                   stroke_axis=2,
                                   the_left_axis=0,
                                   gravity=np.array([0, 0, -9.8]),
                                   sleep_time=0.000000001,
                                   max_angle_of_rotate=pi / 4,
                                   max_angle_of_stroke=pi / 2,
                                   max_force=1000,
                                   max_joint_velocity=100000,
                                   initial_xyz=np.array([0, 0, 0.5]),
                                   initial_rpy=np.array([0, 0, 0]),
                                   right_stroke_joint=0,
                                   right_rotate_joint=1,
                                   left_stroke_joint=2,
                                   left_rotate_joint=3,
                                   right_rod_link=0,
                                   right_wing_link=1,
                                   left_rod_link=2,
                                   left_wing_link=3,
                                   position_gain=0.4,
                                   velocity_gain=0.1, )
