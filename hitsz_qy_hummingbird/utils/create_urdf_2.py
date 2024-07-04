import os.path

import pandas as pd
import xml.etree.ElementTree as ET
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.metrics import r2_score
import pickle
import numpy as np
from hitsz_qy_hummingbird.configuration.configuration import GLOBAL_CONFIGURATION
from sympy.core.mul import Mul
from sympy.solvers import solve
from sympy import Symbol

class URDFCreator:

    def __init__(self,
                 gear_ratio,
                 aspect_ratio,
                 taper_ratio,
                 r22,):
        self.gear_ratio = gear_ratio
        self.aspect_ratio=aspect_ratio
        self.taper_ratio=taper_ratio
        self.r22=r22

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

                self.r = float(sols[R])
                self.chord_root = float(sols[CR])
                self.chord_tip = float(sols[CT])

        self.scaler_wing = None
        self.linear_regression_for_wing = None
        self.scaler_rod = None
        self.linear_regression_for_rod = None

        self.model_path = GLOBAL_CONFIGURATION.urdf_folder_path + 'model_2.pkl'
        if os.path.exists(self.model_path):
            self.load_model()
        else:
            self.dump_model()
        self.urdf_name = None

    def dump_model(self):
        filepath = GLOBAL_CONFIGURATION.urdf_folder_path
        wing_df = pd.read_csv(filepath + "wing24.csv")
        wing_x = wing_df[['R', 'R2', 'R3', 'CT', 'CT2', 'CT3', 'CR', 'CR2', 'CR3', 'RCT', 'RCR', 'CTCR']]
        wing_y = wing_df[['mass', 'Ixx', 'Iyy', 'Izz', 'Iyz']]

        scaler_wing_x = StandardScaler().fit(wing_x)
        standard_wing_x = scaler_wing_x.transform(wing_x)
        # print(scaler_wing_x.get_params())

        linear_regression_for_wing = linear_model.LinearRegression().fit(standard_wing_x, wing_y)
        # print(linear_regression_for_wing.get_params())

        predict_wing_y = linear_regression_for_wing.predict(standard_wing_x)
        print("The w  R_Square is: ", r2_score(wing_y, predict_wing_y))

        rod_df = pd.read_csv(filepath + 'rod24.csv')
        rod_x = rod_df[['R', 'R2', 'R3', 'R4', 'Ng', 'Ng2', 'Ng3', 'Ng4']]
        rod_y = rod_df[['mass', 'Ixx', 'Iyy', 'Izz', 'Iyz']]

        scaler_rod_x = StandardScaler().fit(rod_x)
        standard_rod_x = scaler_rod_x.transform(rod_x)
        # print(scaler_rod_x.get_params())

        linear_regression_for_rod = linear_model.LinearRegression().fit(standard_rod_x, rod_y)
        # print(linear_regression_for_rod.get_params())

        predict_rod_y = linear_regression_for_rod.predict(standard_rod_x)
        print("The r R_Square is: ", r2_score(rod_y, predict_rod_y))

        self.scaler_wing = scaler_wing_x
        self.scaler_rod = scaler_rod_x
        self.linear_regression_for_wing = linear_regression_for_wing
        self.linear_regression_for_rod = linear_regression_for_rod

        with open(filepath + 'model_2.pkl', 'wb') as f:
            pickle.dump(scaler_wing_x, f)
            pickle.dump(linear_regression_for_wing, f)
            pickle.dump(scaler_rod_x, f)
            pickle.dump(linear_regression_for_rod, f)

    def load_model(self):
        with open(GLOBAL_CONFIGURATION.urdf_folder_path + "model_2.pkl", 'rb') as f:
            self.scaler_wing = pickle.load(f)
            self.linear_regression_for_wing = pickle.load(f)
            self.scaler_rod = pickle.load(f)
            self.linear_regression_for_rod = pickle.load(f)

    def write_the_urdf(self):
        '''
        the body inertia and the body mass
        '''

        self.urdf_name = f"{GLOBAL_CONFIGURATION.temporary_urdf_path}Ng_{self.gear_ratio}_AR_{self.aspect_ratio}_TR_{self.taper_ratio}_R22_{self.r22}.urdf"
        print('\n\n\n\n\n')
        print(f'R_{self.r}_CR_{self.chord_root}_CT_{self.chord_tip}.urdf')
        if os.path.exists(self.urdf_name):
            return self.urdf_name


        r = self.r * 1000
        ct = self.chord_tip * 1000
        cr = self.chord_root * 1000
        ng = self.gear_ratio

        wing_x = [[r, r * r, r * r * r, ct, ct * ct, ct * ct * ct, cr, cr * cr, cr * cr * cr, r * ct, r * cr, ct * cr]]
        standard_wing_x = self.scaler_wing.transform(wing_x)
        predicted_wing_y = self.linear_regression_for_wing.predict(standard_wing_x)
        wm = str(predicted_wing_y[0][0] * 1e-6)
        wxx = str(predicted_wing_y[0][1] * 1e-9)
        wyy = str(predicted_wing_y[0][2] * 1e-9)
        wzz = str(predicted_wing_y[0][3] * 1e-9)
        wyz = str(predicted_wing_y[0][4] * 1e-9)
        n_wyz = str(-predicted_wing_y[0][4] * 1e-9)
        # print(wxz)

        rod_x = [[r, r * r, r * r * r, r * r * r * r, ng, ng * ng, ng * ng * ng, ng * ng * ng * ng]]
        standard_rod_x = self.scaler_rod.transform(rod_x)
        predicted_rod_y = self.linear_regression_for_rod.predict(standard_rod_x)
        # print(predicted_rod_y[0])
        rm = str(predicted_rod_y[0][0] * 1e-6)
        rxx = str(predicted_rod_y[0][1] * 1e-9)
        ryy = str(predicted_rod_y[0][2] * 1e-9)
        rzz = str(predicted_rod_y[0][3] * 1e-9)
        ryz = str(predicted_rod_y[0][4] * 1e-9)
        n_ryz = str(-predicted_rod_y[0][4] * 1e-9)
        # print(rm)

        tree = ET.parse(GLOBAL_CONFIGURATION.urdf_folder_path + 'symme0324.urdf')

        root = tree.getroot()
        lr_mass = root.find('./link[@name="lr"]/inertial/mass')
        lr_mass.attrib["value"] = rm
        lr_iner = root.find('./link[@name="lr"]/inertial/inertia')
        lr_iner.attrib["ixx"] = rxx
        lr_iner.attrib["iyy"] = ryy
        lr_iner.attrib["izz"] = rzz
        lr_iner.attrib["iyz"] = ryz

        rr_mass = root.find('./link[@name="rr"]/inertial/mass')
        rr_mass.attrib["value"] = rm
        rr_iner = root.find('./link[@name="rr"]/inertial/inertia')
        rr_iner.attrib["ixx"] = rxx
        rr_iner.attrib["iyy"] = ryy
        rr_iner.attrib["izz"] = rzz
        rr_iner.attrib["iyz"] = n_ryz

        lw_mass = root.find('./link[@name="lw"]/inertial/mass')
        lw_mass.attrib["value"] = wm
        lw_iner = root.find('./link[@name="lw"]/inertial/inertia')
        lw_iner.attrib["ixx"] = wxx
        lw_iner.attrib["iyy"] = wyy
        lw_iner.attrib["izz"] = wzz
        lw_iner.attrib["iyz"] = wyz

        rw_mass = root.find('./link[@name="rw"]/inertial/mass')
        rw_mass.attrib["value"] = wm
        rw_iner = root.find('./link[@name="rw"]/inertial/inertia')
        rw_iner.attrib["ixx"] = wxx
        rw_iner.attrib["iyy"] = wyy
        rw_iner.attrib["izz"] = wzz
        rw_iner.attrib["iyz"] = n_wyz

        tree.write(self.urdf_name)
        return self.urdf_name
