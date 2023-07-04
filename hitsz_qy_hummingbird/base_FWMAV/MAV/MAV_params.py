class ParamsForBaseMAV:
    """
    The Params class is designed to reduce repetitive tasks in future work.
    """

    def __init__(self,
                 rotate_axis: int,
                 stroke_axis: int,
                 the_left_axis: int,
                 gravity: tuple,
                 timestep: int,
                 sleep_time: float,
                 max_angle_of_rotate: float,
                 max_angle_of_stroke: float,
                 max_force: float,
                 max_joint_velocity: float,
                 initial_xyz: tuple,
                 initial_rpy: tuple,
                 right_stroke_joint: int,
                 left_stroke_joint: int,
                 right_rotate_joint: int,
                 left_rotate_joint: int,
                 right_rod_link: int,
                 left_rod_link: int,
                 right_wing_link: int,
                 left_wing_link: int,
                 position_gain: float,
                 velocity_gain: float):
        self.rotate_axis = rotate_axis
        self.stroke_axis = stroke_axis
        self.the_left_axis = the_left_axis
        self.gravity = gravity
        self.timestep = timestep
        self.sleep_time = sleep_time
        self.max_angle_of_rotate = max_angle_of_rotate
        self.max_angle_of_stroke = max_angle_of_stroke
        self.max_force = max_force
        self.initial_xyz = initial_xyz
        self.initial_rpy = initial_rpy
        self.initial_orientation = 0
        self.right_stroke_joint = right_stroke_joint
        self.right_rotate_joint = right_rotate_joint
        self.left_stroke_joint = left_stroke_joint
        self.left_rotate_joint = left_rotate_joint
        self.right_rod_link = right_rod_link
        self.left_rod_link = left_rod_link
        self.right_wing_link = right_wing_link
        self.left_wing_link = left_wing_link
        self.max_joint_velocity = max_joint_velocity
        self.position_gain = position_gain
        self.velocity_gain = velocity_gain