import numpy as np
from sympy.core.mul import Mul
from sympy.solvers import solve
from sympy import Symbol
from math import sin, cos, tan, pi, asin, acos
from scipy import integrate


class ParamsForBaseWing:
    def __init__(self,
                 aspect_ratio,
                 taper_ratio,
                 r22,
                 camber_angle,
                 resolution):

        self.aspect_ratio = aspect_ratio
        self.taper_ratio = taper_ratio
        self.r22 = r22
        self.camber_angle = camber_angle
        self.resolution = resolution
        self.rho_air = 1.1800948
        self.Crd = 5

        R = Symbol('R')
        CT = Symbol("CT")
        CR = Symbol("CR")

        sol = solve([2 * R / (CT + CR) - aspect_ratio,
                     CT / CR - taper_ratio,
                     (R ** 3) * (CT / 4 + CR / 12) - r22], R, CT, CR, dict=True)

        for sols in sol:
            for value in sols.values():
                if isinstance(value, Mul):
                    continue
                if value < 0:
                    continue

                self.length = float(sols[R])
                self.chord_root = float(sols[CR])
                self.chord_tip = float(sols[CT])

        start = self.chord_root * tan(self.camber_angle)
        stop = start + self.length

        chord_space = np.linspace(start=self.chord_root,
                                  stop=self.chord_tip,
                                  num=int(stop * self.resolution))
        r_space = np.linspace(start=- start,
                              stop=stop - start,
                              num=int(stop * self.resolution))
        zero_space = np.linspace(start=0,
                                 stop=0,
                                 num=int(stop * self.resolution))

        chord_line = chord_space[int(start * self.resolution):int(stop * self.resolution)]
        r_line = r_space[int(start * self.resolution):int(stop * self.resolution)]

        def cArBres(c, a, r, b):
            res = pow(c, a) * pow(r, b)
            return integrate.trapz(res, r)

        self.c1 = cArBres(chord_line, 1, r_line, 0)
        self.c2 = cArBres(chord_line, 2, r_line, 0)
        self.c3 = cArBres(chord_line, 3, r_line, 0)
        self.c4 = cArBres(chord_line, 4, r_line, 0)

        self.c2r1 = cArBres(chord_line, 2, r_line, 1)
        self.c2r2 = cArBres(chord_line, 2, r_line, 2)

        self.c1r1 = cArBres(chord_line, 1, r_line, 1)
        self.c1r2 = cArBres(chord_line, 1, r_line, 2)
        self.c1r3 = cArBres(chord_line, 1, r_line, 3)
