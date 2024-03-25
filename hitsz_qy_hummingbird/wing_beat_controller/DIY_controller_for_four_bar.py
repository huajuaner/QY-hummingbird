from sympy import symbols, sin, cos, asin, diff, pi
from hitsz_qy_hummingbird.configuration.configuration import GLOBAL_CONFIGURATION
from enum import Enum


class DIY_controller_for_Four_Bar:
    # WARNING: the returned target velocity is not continuous
    def __init__(self,
                 nominal_amplitude=0,
                 frequency=0,
                 differential_frequency=0,
                 right_split_cycle=0.5,
                 left_split_cycle=0.5,
                 bias = 0):

        self.bias = bias

        self.t, self.An, self.fre, self.FreDiff, self.RightSplit, self.LeftSplit = symbols(
            't An fre FreDiff RightSplit LeftSplit')

        self.right_expr1 = self.An * cos((self.fre + self.FreDiff) * self.t * pi / self.RightSplit)
        self.right_expr2 = self.An * cos(((self.fre + self.FreDiff) * self.t - 1) * pi / (1 - self.RightSplit))

        self.left_expr1 = -self.An * cos((self.fre - self.FreDiff) * self.t * pi / self.LeftSplit)
        self.left_expr2 = -self.An * cos(
            ((self.fre - self.FreDiff) * self.t - 1) * pi / (1 - self.LeftSplit))

        # Differentiate the expressions

        self.right_expr1_diff1 = diff(self.right_expr1, self.t)
        self.right_expr2_diff1 = diff(self.right_expr2, self.t)
        self.left_expr1_diff1 = diff(self.left_expr1, self.t)
        self.left_expr2_diff1 = diff(self.left_expr2, self.t)

        # Initialize default values for parameters
        self.nominal_amplitude = nominal_amplitude
        self.frequency = frequency
        self.differential_frequency = differential_frequency
        self.right_split_cycle = right_split_cycle
        self.left_split_cycle = left_split_cycle

        # Initialize substituted expressions
        self.right_expr1_sub = None
        self.right_expr2_sub = None
        self.right_expr1_sub_diff1 = None
        self.right_expr2_sub_diff1 = None
        self.left_expr1_sub = None
        self.left_expr2_sub = None
        self.left_expr1_sub_diff1 = None
        self.left_expr2_sub_diff1 = None

        self.data = {
            'target_right_stroke_amp_lis': [],
            'target_right_stroke_vel_lis': [],
            'target_left_stroke_amp_lis': [],
            'target_left_stroke_vel_lis': [],
            'seeseelist': []
        }
        self.update()

    def change_parameter(self,
                         nominal_amplitude=None,
                         frequency=None,
                         differential_frequency=None,
                         right_split_cycle=None,
                         left_split_cycle=None):

        if nominal_amplitude is not None:
            self.nominal_amplitude = nominal_amplitude
        if frequency is not None:
            self.frequency = frequency
        if differential_frequency is not None:
            self.differential_frequency = differential_frequency
        if right_split_cycle is not None:
            self.right_split_cycle = right_split_cycle
        if left_split_cycle is not None:
            self.left_split_cycle = left_split_cycle
        self.update()

    def update(self):
        mydict = [(self.An, self.nominal_amplitude),
                  (self.fre, self.frequency),
                  (self.FreDiff, self.differential_frequency),
                  (self.RightSplit, self.right_split_cycle),
                  (self.LeftSplit, self.left_split_cycle),
                  (pi, 3.14159265)]

        self.right_expr1_sub = self.right_expr1.subs(mydict)
        self.right_expr1_sub_diff1 = self.right_expr1_diff1.subs(mydict)
        self.right_expr2_sub = self.right_expr2.subs(mydict)
        self.right_expr2_sub_diff1 = self.right_expr2_diff1.subs(mydict)

        self.left_expr1_sub = self.left_expr1.subs(mydict)
        self.left_expr1_sub_diff1 = self.left_expr1_diff1.subs(mydict)
        self.left_expr2_sub = self.left_expr2.subs(mydict)
        self.left_expr2_sub_diff1 = self.left_expr2_diff1.subs(mydict)

        print(self.right_expr1_sub)
        print(self.right_expr1_sub_diff1)
        print(self.right_expr2_sub)
        print(self.right_expr2_sub_diff1)

        print(self.left_expr1_sub)
        print(self.left_expr1_sub_diff1)
        print(self.left_expr2_sub)
        print(self.left_expr2_sub_diff1)

    def step(self):
        """
        Use the universal time in GLOBAL_CONFIGURATION
        Return the rightTargetPos, rightTargetVel, leftTargetPos, LeftTargetVel
        """
        the_time = GLOBAL_CONFIGURATION.TIME

        if GLOBAL_CONFIGURATION.TIME < 0.10:
            k_a = GLOBAL_CONFIGURATION.TIME / 0.10
        else:
            k_a = 1

        # k_a = 1

        aa = the_time * self.frequency % 1
        self.data['seeseelist'].append(aa)

        time = the_time % (1 / (self.frequency + self.differential_frequency))
        print("******")
        print(time)
        mydict = [(self.t, time)]
        if (time * (self.frequency + self.differential_frequency)) % 1 < self.right_split_cycle:
            right_amp = self.right_expr1_sub.subs(mydict) * k_a
            right_vel = self.right_expr1_sub_diff1.subs(mydict) * k_a
        else:
            right_amp = self.right_expr2_sub.subs(mydict) * k_a
            right_vel = self.right_expr2_sub_diff1.subs(mydict) * k_a

        the_time = the_time - self.bias * 3.14 / (self.frequency - self.differential_frequency)
        time = the_time % (1 / (self.frequency - self.differential_frequency))
        print(time)
        mydict = [(self.t, time)]
        if (time * (self.frequency - self.differential_frequency)) % 1 < self.left_split_cycle:
            left_amp = self.left_expr1_sub.subs(mydict) * k_a
            left_vel = self.left_expr1_sub_diff1.subs(mydict) * k_a
        else:
            left_amp = self.left_expr2_sub.subs(mydict) * k_a
            left_vel = self.left_expr2_sub_diff1.subs(mydict) * k_a

        self.data['target_right_stroke_amp_lis'].append(right_amp)
        self.data['target_right_stroke_vel_lis'].append(right_vel)

        self.data['target_left_stroke_amp_lis'].append(left_amp)
        self.data['target_left_stroke_vel_lis'].append(left_vel)

        return [left_amp,
                left_vel,
                right_amp,
                right_vel]
