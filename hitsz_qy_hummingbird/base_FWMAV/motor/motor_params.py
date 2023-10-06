class ParamsForBaseMotor:
    def __init__(self,
                 **kwargs):
        self.terminal_r = None
        self.terminal_l = None
        self.torque_constant = None
        self.speed_constant = None
        self.rotor_inertia = None
        self.nominal_current = None
        self.nominal_voltage = None
        self.rotor_damping_coeff = None
        self.wing_damping_coeff = None

        self.gear_efficiency = None
        self.gear_ratio = None

        self.spring_youngs_modulus = 193
        self.spring_wire_diameter = None
        self.spring_number_of_coils = None
        self.spring_outer_diameter = None

        self.spring_constant = None
        self.change_parameters(**kwargs)

    def change_parameters(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"Invalid parameter '{key}'")

        # Recalculate the spring constant if any relevant parameters have been modified
        if any(parameter in kwargs for parameter in
               ["spring_youngs_modulus", "spring_wire_diameter", "spring_number_of_coils", "spring_outer_diameter"]):
            self.calculate_spring_constant()

    def calculate_spring_constant(self):
        if all(
                hasattr(self, attr)
                for attr in
                ["spring_youngs_modulus", "spring_wire_diameter", "spring_number_of_coils", "spring_outer_diameter"]
        ):
            self.spring_constant = 2 * self.spring_wire_diameter ** 4 \
                                   / (67 * self.spring_number_of_coils * self.spring_outer_diameter) * 100
            """
            The Calculation is K_s = E * d ^ 4 / ( 64 * outer_diameter * number_of_winding)
            For Purdue Univ, E is 193Gpa which is 193 for calculation
            wire diameter d is 0.3mm 
            outer diameter D is 2.67mm 
            number of winding n is 4.25 
            The k_s is 0.0022 Nm/rad  
            """

        else:
            raise ValueError("Cannot calculate spring constant. Missing parameters.")

    def valid_check(self):
        attributes = [
            "terminal_r", "terminal_l", "torque_constant", "speed_constant", "rotor_inertia",
            "rotor_damping_coeff", "gear_efficiency", "gear_ratio", "wing_damping_coeff",
            "spring_youngs_modulus", "spring_wire_diameter", "spring_number_of_coils",
            "spring_outer_diameter", "spring_constant", "nominal_current", "nominal_voltage"
        ]

        for attr in attributes:
            if getattr(self, attr) is None:
                raise ValueError(f"Parameter '{attr}' is missing")
        return True
