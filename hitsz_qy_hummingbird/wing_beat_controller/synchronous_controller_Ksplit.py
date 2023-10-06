from hitsz_qy_hummingbird.configuration.configuration import GLOBAL_CONFIGURATION
from enum import Enum
from sympy import symbols, sin, cos, asin, diff, pi


class SynchronousControllerKsplit:

    def __init__(self,
                 nominal_amplitude=0,
                 frequency=0,
                 differential_amplitude=0,
                 bias_amplitude=0,
                 split_cycle=0.5):

        self.nominal_amplitude = nominal_amplitude
        self.frequency = frequency
        self.differential_amplitude = differential_amplitude
        self.bias_amplitude = bias_amplitude
        self.split_cycle = split_cycle

        self.t, self.An, self.fre, self.Adiff, self.Abias, self.Ksplit = symbols('t An fre Adiff Abias Ksplit')
        self.right_expr1 = (self.An + self.Adiff) * sin(pi * self.fre * self.t / self.Ksplit) + self.Abias
        self.right_expr2 = (self.An + self.Adiff) * sin((pi * self.fre * self.t - pi) / (1 - self.Ksplit)) + self.Abias
        self.left_expr1 = (self.An - self.Adiff) * sin(pi * self.fre * self.t / self.Ksplit) + self.Abias
        self.left_expr2 = (self.An - self.Adiff) * sin((pi * self.fre * self.t - pi) / (1 - self.Ksplit)) + self.Abias

        self.right_expr1_diff = diff(self.right_expr1, self.t)
        self.right_expr2_diff = diff(self.right_expr2, self.t)
        self.left_expr1_diff = diff(self.left_expr1, self.t)
        self.left_expr2_diff = diff(self.left_expr2, self.t)

        self.right_expr1_sub = None
        self.right_expr2_sub = None
        self.right_expr1_diff_sub = None
        self.right_expr2_diff_sub = None
        self.left_expr1_sub = None
        self.left_expr2_sub = None
        self.left_expr1_diff_sub = None
        self.left_expr2_diff_sub = None

        self.update()

    def change_parameter(self,
                         nominal_amplitude=None,
                         frequency=None,
                         differential_amplitude=None,
                         bias_amplitude=None,
                         split_cycle=None):

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

        self.update()

    def update(self):
        self.mydict = [(self.An, self.nominal_amplitude),
                       (self.fre, self.frequency),
                       (self.Adiff, self.differential_amplitude),
                       (self.Abias, self.bias_amplitude),
                       (self.Ksplit, self.split_cycle),
                       (pi, 3.14159265)]
        self.right_expr1_sub = self.right_expr1.subs(self.mydict)
        self.right_expr2_sub = self.right_expr2.subs(self.mydict)
        self.right_expr1_diff_sub = self.right_expr1_diff.subs(self.mydict)
        self.right_expr2_diff_sub = self.right_expr2_diff.subs(self.mydict)

        self.left_expr1_sub = self.left_expr1.subs(self.mydict)
        self.left_expr2_sub = self.left_expr2.subs(self.mydict)
        self.left_expr1_diff_sub = self.left_expr1_diff.subs(self.mydict)
        self.left_expr2_diff_sub = self.left_expr2_diff.susb(self.mydict)

    def step(self):
        the_time = GLOBAL_CONFIGURATION.TIME
        time = the_time % (1 / self.frequency)
        mydict = [(self.t, time)]

        if (time * self.frequency) % 1 < self.split_cycle:
            return [self.right_expr1_sub.subs(mydict),
                    self.right_expr1_diff_sub.subs(mydict),
                    self.left_expr1_sub.subs(mydict),
                    self.left_expr1_diff_sub.subs(mydict)]
        else:
            return [self.right_expr2_sub.subs(mydict),
                    self.right_expr2_diff_sub.subs(mydict),
                    self.left_expr2_sub.subs(mydict),
                    self.left_expr2_diff_sub.subs(mydict)]
