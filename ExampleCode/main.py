from hitsz_qy_hummingbird.base_FWMAV.motor.motor_params import ParamsForBaseMotor

motor = ParamsForBaseMotor(spring_wire_diameter=0.7,
                           spring_number_of_coils=6,
                           spring_outer_diameter=3.5)
print(motor.spring_constant)