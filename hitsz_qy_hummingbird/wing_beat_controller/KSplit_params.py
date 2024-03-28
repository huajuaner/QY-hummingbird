from hitsz_qy_hummingbird.wing_beat_controller.synchronous_controller_Ksplit import SynchronousControllerKsplit


class ParamsForKSplit:
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
        self.controller = SynchronousControllerKsplit(nominal_amplitude=self.nominal_amplitude,
                                                      frequency=self.frequency,
                                                      differential_amplitude=self.differential_amplitude,
                                                      bias_amplitude=self.bias_amplitude,
                                                      split_cycle=self.split_cycle)

    def the_according_controller(self):
        return self.controller
