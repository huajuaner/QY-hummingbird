from sympy import symbols, sin, cos, asin, diff, pi
from sympy.plotting import plot


class WingBeatProfile:
    def __init__(self):

        self.t, self.An, self.fre, self.Adiff, self.Abias, self.Ksplit, self.Ksquare = symbols(
            't An fre Adiff Abias Ksplit Ksquare')

        # Right wing expression
        self.right_expr1 = self.Abias + (self.An + self.Adiff) / asin(self.Ksquare) \
                           * asin(self.Ksquare * cos(pi * self.fre * self.t / self.Ksplit))
        self.right_expr2 = self.Abias + (self.An + self.Adiff) / asin(self.Ksquare) \
                           * asin(
            self.Ksquare * cos((self.fre * pi * self.t + pi - 2 * pi * self.Ksplit) / (1 - self.Ksplit)))

        # Left wing expression
        self.left_expr1 = self.Abias + (self.An - self.Adiff) / asin(self.Ksquare) \
                          * asin(self.Ksquare * cos(pi * self.fre * self.t / self.Ksplit))
        self.left_expr2 = self.Abias + (self.An - self.Adiff) / asin(self.Ksquare) \
                          * asin(
            self.Ksquare * cos((self.fre * pi * self.t + pi - 2 * pi * self.Ksplit) / (1 - self.Ksplit)))

        # Differentiate the expressions
        self.right_expr1_diff1 = diff(self.right_expr1, self.t)
        self.right_expr1_diff2 = diff(self.right_expr1, self.t, 2)

        self.right_expr2_diff1 = diff(self.right_expr2, self.t)
        self.right_expr2_diff2 = diff(self.right_expr2, self.t, 2)

        self.left_expr1_diff1 = diff(self.left_expr1, self.t)
        self.left_expr1_diff2 = diff(self.left_expr1, self.t, 2)

        self.left_expr2_diff1 = diff(self.left_expr2, self.t)
        self.left_expr2_diff2 = diff(self.left_expr2, self.t, 2)

        # Initialize default values for parameters
        self.nominal_amplitude = 0
        self.frequency = 0
        self.differential_amplitude = 0
        self.bias_amplitude = 0
        self.split_cycle = 0.5
        self.square_parameter = 0

        # Initialize substituted expressions
        self.right_expr1_sub = None
        self.right_expr2_sub = None
        self.right_expr1_sub_diff1 = None
        self.right_expr2_sub_diff1 = None
        self.right_expr1_sub_diff2 = None
        self.right_expr2_sub_diff2 = None

        self.left_expr1_sub = None
        self.left_expr2_sub = None
        self.left_expr1_sub_diff1 = None
        self.left_expr2_sub_diff1 = None
        self.left_expr1_sub_diff2 = None
        self.left_expr2_sub_diff2 = None

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

        mydict = [(self.An, self.nominal_amplitude),
                  (self.fre, self.frequency),
                  (self.Adiff, self.differential_amplitude),
                  (self.Abias, self.bias_amplitude),
                  (self.Ksplit, self.split_cycle),
                  (self.Ksquare, self.square_parameter),
                  (pi, 3.1414926)]

        self.right_expr1_sub = self.right_expr1.subs(mydict)
        self.right_expr1_sub_diff1 = self.right_expr1_diff1.subs(mydict)
        self.right_expr1_sub_diff2 = self.right_expr1_diff2.subs(mydict)

        self.right_expr2_sub = self.right_expr2.subs(mydict)
        self.right_expr2_sub_diff1 = self.right_expr2_diff1.subs(mydict)
        self.right_expr2_sub_diff2 = self.right_expr2_diff2.subs(mydict)

        self.left_expr1_sub = self.left_expr1.subs(mydict)
        self.left_expr1_sub_diff1 = self.left_expr1_diff1.subs(mydict)
        self.left_expr1_sub_diff2 = self.left_expr1_diff2.subs(mydict)

        self.left_expr2_sub = self.left_expr2.subs(mydict)
        self.left_expr2_sub_diff1 = self.left_expr2_diff1.subs(mydict)
        self.left_expr2_sub_diff2 = self.left_expr2_diff2.subs(mydict)

    def step(self, time):
        """
        Input the time in s
        Return the rightTargetPos, rightTargetVel, rightTargetAcc, leftTargetPos, LeftTargetVel, LeftTargetAcc
        """
        time = time % (1 / self.frequency)
        mydict = [(self.t, time)]

        if (time * self.frequency) % 1 < self.split_cycle:
            return [self.right_expr1_sub.subs(mydict),
                    self.right_expr1_sub_diff1.subs(mydict),
                    self.right_expr1_sub_diff2.subs(mydict),
                    self.left_expr1_sub.subs(mydict),
                    self.left_expr1_sub_diff1.subs(mydict),
                    self.left_expr1_sub_diff2.subs(mydict)]
        else:
            return [self.right_expr2_sub.subs(mydict),
                    self.right_expr2_sub_diff1.subs(mydict),
                    self.right_expr2_sub_diff2.subs(mydict),
                    self.left_expr2_sub.subs(mydict),
                    self.left_expr2_sub_diff1.subs(mydict),
                    self.left_expr2_sub_diff2.subs(mydict)]
