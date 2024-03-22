
class WingBeatTorque:
    
    def __init__(self):
        self.right_torque =0
        self.left_torque =0.01
    
    def step(self):
        return self.right_torque, self.left_torque