import numpy as np

class BaseMAV:
    def __init__(self):
        pass

    def change_joint_dynamics(self):
        pass

    def camera_follow(self):
        pass

    def step(self):
        pass

    def joint_state_update(self):
        pass

    def joint_control(self,
                      target_right_stroke_amp=None,
                      target_right_stroke_vel=None,
                      target_left_stroke_amp=None,
                      target_left_stroke_vel=None,
                      right_input_torque=None,
                      left_input_torque=None):
        pass

    def get_state_for_motor_torque(self):
        pass

    def get_state_for_wing(self):
        pass

    def set_link_force_world_frame(self,
                                   link_id,
                                   position_bias: np.ndarray,
                                   force: np.ndarray):
        pass

    def set_link_torque_world_frame(self,
                                    linkid,
                                    torque: np.ndarray):
        pass

    def draw_a_line(self,
                    start,
                    end,
                    line_color,
                    name):
        pass

    def get_constraint_state(self):
        pass

    def reset(self):
        pass

    def close(self):
        pass

    def housekeeping(self):
        pass
