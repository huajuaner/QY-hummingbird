import numpy as np
from sympy.core.mul import Mul
from sympy.solvers import solve
from sympy import Symbol
from math import sin, cos, tan, pi, asin, acos
from scipy import integrate
from hitsz_qy_hummingbird.configuration.configuration import GLOBAL_CONFIGURATION
from hitsz_qy_hummingbird.base_FWMAV.wings.wing_params import ParamsForBaseWing


class BaseWing:
    """
    all length variables are in meters
    all weight variables are in kg
    all angular variables are in rad
    notably in the Base Wing Class only the nominal force is considered
    """

    def __init__(self,
                 params: ParamsForBaseWing):
        self.params = params
        self.logger = GLOBAL_CONFIGURATION.logger
        self.logger.debug(f"the length is {self.params.length}")
        self.logger.debug(f"the chord of root is {self.params.chord_root}")
        self.logger.debug(f"the chord of tip is {self.params.chord_tip}")
        self.data={}

    def calculate_aeroforce_and_torque(self,
                                       stroke_angular_velocity: np.ndarray,
                                       rotate_angular_velocity: np.ndarray,
                                       r_axis: np.ndarray,
                                       c_axis: np.ndarray,
                                       z_axis: np.ndarray):
        """
        :param stroke_angular_velocity:
        :param rotate_angular_velocity:
        :param r_axis:
        :param c_axis:
        :param z_axis:
        :return the aeroforce,forcepos and aerotorque:
        """
        if self.definite_value(stroke_angular_velocity) + self.definite_value(rotate_angular_velocity) == 0:
            return [0, 0, 0], [0, 0, 0], [0, 0, 0]

        vel = np.cross(stroke_angular_velocity, r_axis)
        angle_of_attack = self.the_angle_between(vel, c_axis)
        if angle_of_attack > pi / 2:
            angle_of_attack = pi - angle_of_attack

        stroke_vel = self.definite_value(stroke_angular_velocity)
        rotate_vel = self.definite_value(rotate_angular_velocity)

        coeff = 1 / 2 * self.params.rho_air * stroke_vel * stroke_vel * self.CN(angle_of_attack)
        aeroforce = coeff * self.params.c1r2
        torque_c_wise = coeff * self.params.c2r2
        torque_r_wise = coeff * self.params.c1r3
        forcepos = r_axis * torque_r_wise / aeroforce + c_axis * torque_c_wise / aeroforce
        aerotorque = 1 / 8 * self.params.rho_air * self.params.Crd * self.params.c4 * rotate_vel * rotate_vel
        if rotate_angular_velocity.dot(r_axis) > 0 :
            aerotorque = aerotorque * -1
        aerotorque = aerotorque * r_axis
        aeroforce = aeroforce * z_axis

        return aeroforce, forcepos, aerotorque

    @staticmethod
    def CL(angle_of_attack):
        return 1.8 * sin(2 * angle_of_attack)

    @staticmethod
    def CD(angle_of_attack):
        return 1.9 - 1.5 * cos(2 * angle_of_attack)

    def CN(self, angle_of_attack):
        return cos(angle_of_attack) * self.CL(angle_of_attack) \
            + sin(angle_of_attack) * self.CD(angle_of_attack)

    @staticmethod
    def definite_value(vec: np.ndarray):
        return np.sqrt(vec.dot(vec))

    @staticmethod
    def dcp(angle_of_attack):
        return 0.82 * angle_of_attack / pi + 0.05

    @staticmethod
    def new_dcp(angle_of_attack):
        """
        This is inherited from Purdue University
        """
        return 0.46 - 0.332 * cos(angle_of_attack) \
            - 0.037 * cos(3 * angle_of_attack) \
            - 0.013 * (5 * angle_of_attack)

    def the_angle_between(self,
                          vecA: np.ndarray,
                          vecB: np.ndarray):
        cosvalue = vecA.dot(vecB)
        res = cosvalue / (self.definite_value(vecA) * self.definite_value(vecB))
        return acos(res)

    def housekeeping(self):
        self.data.clear()