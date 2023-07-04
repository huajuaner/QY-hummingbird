class ParamsForBaseMotor:
    def __init__(self,
                 terminal_r,
                 terminal_l,
                 torque_constant,
                 speed_constant,
                 rotor_inertia,
                 rotor_damping_coeff,
                 gear_efficiency,
                 gear_ratio,
                 wing_damping_coeff,
                 spring_youngs_modulus,
                 spring_wire_diameter,
                 spring_number_of_coils,
                 spring_outer_diameter):
        self.terminal_r = terminal_r
        self.terminal_l = terminal_l
        self.torque_constant  = torque_constant
        self.speed_constant = speed_constant
        self.rotor_inertia = rotor_inertia
        self.rotor_damping_coeff = rotor_damping_coeff
        self.gear_efficiency = gear_efficiency
        self.gear_ratio = gear_ratio
        self.wing_damping_coeff = wing_damping_coeff
        self.spring_youngs_modulus = spring_youngs_modulus
        self.spring_wire_diameter = spring_wire_diameter
        self.spring_number_of_coils = spring_number_of_coils
        self.spring_outer_diameter = spring_outer_diameter

        #TODO calculate the according spring constant
        self.spring_constant = 1