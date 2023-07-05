class ParamsForBaseMAV:
    def __init__(self, **kwargs):
        self.rotate_axis = None
        self.stroke_axis = None
        self.the_left_axis = None
        self.gravity = None
        self.sleep_time = None
        self.max_angle_of_rotate = None
        self.max_angle_of_stroke = None
        self.max_force = None
        self.max_joint_velocity = None
        self.initial_xyz = None
        self.initial_rpy = None
        self.initial_orientation = None
        self.right_stroke_joint = None
        self.left_stroke_joint = None
        self.right_rotate_joint = None
        self.left_rotate_joint = None
        self.right_rod_link = None
        self.left_rod_link = None
        self.right_wing_link = None
        self.left_wing_link = None
        self.position_gain = None
        self.velocity_gain = None

        self.change_parameters(**kwargs)

    def change_parameters(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"Invalid parameter '{key}'")

    def valid_check(self):
        attributes = [
            "rotate_axis", "stroke_axis", "the_left_axis", "gravity", "sleep_time",
            "max_angle_of_rotate", "max_angle_of_stroke", "max_force", "max_joint_velocity",
            "initial_xyz", "initial_rpy", "right_stroke_joint", "left_stroke_joint",
            "right_rotate_joint", "left_rotate_joint", "right_rod_link", "left_rod_link",
            "right_wing_link", "left_wing_link", "position_gain", "velocity_gain"
        ]

        for attr in attributes:
            if getattr(self, attr) is None:
                raise AttributeError(f"Parameter '{attr}' is missing")
        return True
