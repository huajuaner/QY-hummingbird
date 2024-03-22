import numpy as np
import pybullet as p
from pybullet_utils import bullet_client
import pybullet_data
import gymnasium as gym
from gymnasium import spaces

class RLMAV(gym.Env):
    """
    This class is specially defined for the test of RL

    """

    def __init__(self,
                 gui = False,
                 ):
        if gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        self.flapper_ID = 0
        self._housekeeping(self.physics_client)

        self.action_space = self._actionSpace()
        self.observation_space = self._observationSpace()

    def step(self,
            action):
        """
        
        """
        flapperPos, flapperOrn = p.getBasePositionAndOrientation(self.flapper_ID)
        #step其余代码
        return obs, reward, terminated, truncated, info
    
    def reset(self,
            seed : int = None,):
        p.resetSimulation(physicsClientId=self.physics_client)
        #### Housekeeping ##########################################
        self._housekeeping(self.physics_client)
        self.pos, self.quat = p.getBasePositionAndOrientation(self.flapper_ID)
        #reset其余代码
        return initial_obs, initial_info
    
    def _housekeeping(self,physics_client):
        #在BaseMAV中进行了加载模型的操作，即self.body_unique_id = p.loadURDF()
        self.mav = BaseMAV(client=physics_client,
                           if_fixed=False)
        self.flapper_ID = self.mav.body_unique_id
