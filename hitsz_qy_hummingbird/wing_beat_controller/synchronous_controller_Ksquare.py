from sympy import symbols, sin, cos, asin, diff, pi
from hitsz_qy_hummingbird.configuration.configuration import GLOBAL_CONFIGURATION
from enum import Enum


class SynchronousControllerKsquare:
    # WARNING: the returned target velocity is not continuous

    def __init__(self,
                 nominal_amplitude=0,
                 frequency=0,
                 differential_amplitude=0,
                 bias_amplitude=0,
                 split_cycle=0.5,
                 square_parameter=0.0001):

        self.t, self.An, self.fre, self.Adiff, self.Abias, self.Ksplit, self.Ksquare = symbols(
            't An fre Adiff Abias Ksplit Ksquare')

        # Right wing expression

        self.right_expr1 = (self.Abias
                            + (self.An + self.Adiff) / asin(self.Ksquare)
                            * asin(self.Ksquare * sin(pi * self.fre * self.t
                                                      / self.Ksplit)))
        self.right_expr2 = (self.Abias
                            + (self.An + self.Adiff) / asin(self.Ksquare)
                            * asin(self.Ksquare * sin((self.fre * pi * self.t + pi - 2 * pi * self.Ksplit)
                                                      / (1 - self.Ksplit))))

        # Left wing expression
        self.left_expr1 = (self.Abias - (self.An - self.Adiff) / asin(self.Ksquare)
                           * asin(self.Ksquare * sin(pi * self.fre * self.t
                                                     / self.Ksplit)))
        self.left_expr2 = (self.Abias - (self.An - self.Adiff) / asin(self.Ksquare)
                           * asin(self.Ksquare * sin((self.fre * pi * self.t + pi - 2 * pi * self.Ksplit)
                                                     / (1 - self.Ksplit))))

        # Differentiate the expressions

        self.right_expr1_diff1 = diff(self.right_expr1, self.t)
        self.right_expr2_diff1 = diff(self.right_expr2, self.t)
        self.left_expr1_diff1 = diff(self.left_expr1, self.t)
        self.left_expr2_diff1 = diff(self.left_expr2, self.t)

        # Initialize default values for parameters
        self.nominal_amplitude = nominal_amplitude
        self.frequency = frequency
        self.differential_amplitude = differential_amplitude
        self.bias_amplitude = bias_amplitude
        self.split_cycle = split_cycle
        self.square_parameter = square_parameter

        # Initialize substituted expressions
        self.right_expr1_sub = None
        self.right_expr2_sub = None
        self.right_expr1_sub_diff1 = None
        self.right_expr2_sub_diff1 = None
        self.left_expr1_sub = None
        self.left_expr2_sub = None
        self.left_expr1_sub_diff1 = None
        self.left_expr2_sub_diff1 = None

        self.update()

    def change_parameter(self,
                         nominal_amplitude: float = None,
                         frequency: int = None,
                         differential_amplitude: float = None,
                         bias_amplitude: float = None,
                         split_cycle: float = None,
                         square_parameter: float = None):
        if nominal_amplitude is not None:
            self.nominal_amplitude = nominal_amplitude
        if frequency is not None:
            self.frequency = frequency
        if differential_amplitude is not None:
            self.differential_amplitude = differential_amplitude
        if bias_amplitude is not None:
            self.bias_amplitude = bias_amplitude
        if split_cycle is not None:
            self.split_cycle = split_cycle
        if square_parameter is not None:
            self.square_parameter = square_parameter

        self.update()

    def update(self):
        mydict = [(self.An, self.nominal_amplitude),
                  (self.fre, self.frequency),
                  (self.Adiff, self.differential_amplitude),
                  (self.Abias, self.bias_amplitude),
                  (self.Ksplit, self.split_cycle),
                  (self.Ksquare, self.square_parameter),
                  (pi, 3.14159265)]

        self.right_expr1_sub = self.right_expr1.subs(mydict)
        self.right_expr1_sub_diff1 = self.right_expr1_diff1.subs(mydict)
        self.right_expr2_sub = self.right_expr2.subs(mydict)
        self.right_expr2_sub_diff1 = self.right_expr2_diff1.subs(mydict)

        self.left_expr1_sub = self.left_expr1.subs(mydict)
        self.left_expr1_sub_diff1 = self.left_expr1_diff1.subs(mydict)
        self.left_expr2_sub = self.left_expr2.subs(mydict)
        self.left_expr2_sub_diff1 = self.left_expr2_diff1.subs(mydict)

    def step(self):
        """
        Use the universal time in GLOBAL_CONFIGURATION
        Return the rightTargetPos, rightTargetVel, leftTargetPos, LeftTargetVel
        """
        the_time = GLOBAL_CONFIGURATION.TIME
        time = the_time % (1 / self.frequency)
        mydict = [(self.t, time)]

        if GLOBAL_CONFIGURATION.TIME < 0.10 :
            k_a = GLOBAL_CONFIGURATION.TIME / 0.10
        else :
            k_a = 1

        # k_a = 1

        if (time * self.frequency) % 1 < self.split_cycle:
            return [self.right_expr1_sub.subs(mydict)*k_a,
                    self.right_expr1_sub_diff1.subs(mydict)*k_a,
                    self.left_expr1_sub.subs(mydict)*k_a,
                    self.left_expr1_sub_diff1.subs(mydict)*k_a]
        else:
            return [self.right_expr2_sub.subs(mydict)*k_a,
                    self.right_expr2_sub_diff1.subs(mydict)*k_a,
                    self.left_expr2_sub.subs(mydict)*k_a,
                    self.left_expr2_sub_diff1.subs(mydict)*k_a]
