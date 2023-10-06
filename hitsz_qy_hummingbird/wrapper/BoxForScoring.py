from hitsz_qy_hummingbird.wrapper.base_wrapped_mav import BaseWrappedMAV
from hitsz_qy_hummingbird.wing_beat_controller.synchronous_controller_Ksplit import SynchronousControllerKsplit

class BoxForScoring:
    '''
    The target base_wrapped_mav should run x periods
    The data can be filtered using the spikey filter
    the original data and the filtered data should be restored
    '''
    def __init__(self,
                 mav:BaseWrappedMAV,
                 controller:SynchronousControllerKsplit,
                 num_of_iterations:int,
                 readme):
        self.mav = mav
        self.controller = controller
        self.num_of_iterations = num_of_iterations
        self.readme = readme

    def Scoring(self):



