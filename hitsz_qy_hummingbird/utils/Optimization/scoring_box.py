from hitsz_qy_hummingbird.wrapper.wrapped_mav_for_design import WrappedMAVDesign
from hitsz_qy_hummingbird.wing_beat_controller.synchronous_controller_Ksplit import SynchronousControllerKsplit
from hitsz_qy_hummingbird.configuration.configuration import GLOBAL_CONFIGURATION
from hitsz_qy_hummingbird.base_FWMAV.MAV import MAV_params
from hitsz_qy_hummingbird.base_FWMAV.MAV.base_MAV_sequential import BaseMavSequential
from hitsz_qy_hummingbird.base_FWMAV.motor.motor_params import ParamsForBaseMotor
from hitsz_qy_hummingbird.base_FWMAV.wings.wing_params import ParamsForBaseWing
from tqdm import tqdm
import numpy as np

class BoxForScoring:
    """
    The target base_wrapped_mav should run x periods
    The data can be filtered using the spikey filter
    the original data and the filtered data should be restored
    """

    def __init__(self,
                 mav_params: MAV_params,
                 motor_params: ParamsForBaseMotor,
                 wing_params: ParamsForBaseWing,
                 controller: SynchronousControllerKsplit,
                 periods=GLOBAL_CONFIGURATION.TIMESTEP,
                 beginning  = 0 ,
                 ending = 0 ):
        self.urdf_name = GLOBAL_CONFIGURATION.urdf_folder_path + "symme0324.urdf"
        self.mav = BaseMavSequential(urdf_name=self.urdf_name,
                                     mav_params=mav_params,
                                     if_gui=False,
                                     if_fixed=True)
        self.wrapped_mav = WrappedMAVDesign(mav=self.mav,
                                            motor_params=motor_params,
                                            wing_params=wing_params)
        self.controller = controller
        self.periods = periods
        self.beginning = int(self.periods/2)
        self.ending = self.periods
        self.data = {
            "right_power": [],
            "right_voltage": [],
            "right_stroke_amp": [],
            "right_torque": [],
            "left_power": [],
            "left_voltage": [],
            "left_stroke_amp": [],
            "left_torque": [],
            "Force1": [],
            "Force2": [],
            "LiftForce": [],
            "Torque1": [],
            "Torque2": [],
            "Torque3": []
        }

    def run(self):
        """
        completing each period of steps,
        and the data is stored
        """
        GLOBAL_CONFIGURATION.timereset()
        with tqdm(total=self.periods) as pbar:
            for i in range(0, self.periods):
                if not i % 200:
                    pbar.update(200)
                right_voltage, left_voltage = self.controller.simple_step()
                # print(right_voltage, left_voltage)
                right_torque,left_torque = self.wrapped_mav.joint_control_voltage((right_voltage, -1 * left_voltage))
                right_stroke_amp, _, _, _, right_i, right_v, left_stroke_amp, _, _, _, left_i, left_v, f1, f2, f3, t1, t2, t3 = self.wrapped_mav.step_after_joint_control()
                self.data["right_power"].append(right_i * right_v)
                self.data["right_voltage"].append(right_v)
                self.data["right_stroke_amp"].append(right_stroke_amp)
                self.data["right_torque"].append(right_torque)
                self.data["left_power"].append(left_i * left_v)
                self.data["left_voltage"].append(left_v)
                self.data["left_stroke_amp"].append(left_stroke_amp)
                self.data["left_torque"].append(left_torque)
                self.data["Force1"].append(f1)
                self.data["Force2"].append(f2)
                self.data["LiftForce"].append(-1*f3)
                self.data["Torque1"].append(t1)
                self.data["Torque2"].append(t2)
                self.data["Torque3"].append(t3)
                GLOBAL_CONFIGURATION.step()

    def control_authority(self):
        pass

    def power_loading_scoring(self):

        LiftForce = np.array(self.data["LiftForce"][self.beginning:self.ending]).mean()
        PowerConsumption = np.array(self.data["right_power"][self.beginning:self.ending]).mean()
        print(LiftForce)
        print(PowerConsumption)
        return LiftForce/PowerConsumption

    def close(self):
        self.wrapped_mav.close()
