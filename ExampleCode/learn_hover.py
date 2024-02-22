import sys 
sys.path.append('D:/graduate/fwmav/Simul2023/230818/QY-hummingbird-main')
from hitsz_qy_hummingbird.envs.RL_wrapped import RLMAV
from hitsz_qy_hummingbird.configuration import configuration
from hitsz_qy_hummingbird.base_FWMAV.wings.wing_params import ParamsForBaseWing

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

class learn_hover:

    def __init__(self):
        self.motor_params = configuration.ParamsForMaxonSpeed6M
        self.motor_params.change_parameters(spring_wire_diameter=0.5,
                               spring_number_of_coils=6,
                               spring_outer_diameter=3.5,
                               gear_efficiency=0.8,
                               gear_ratio=10)

        self.wing_params = ParamsForBaseWing(aspect_ratio=9.3,
                                     taper_ratio=0.66,
                                r22=4E-6,
                                camber_angle=16 / 180 * np.pi,
                                resolution=500)

        self.mav_params = configuration.ParamsForMAV_One.change_parameters(sleep_time=0.000001)
    
    def run(self):

        #### Check the environment's spaces ########################
        env = gym.make("hover-qyhb-v0")
        print("[INFO] Action space:", env.action_space)
        print("[INFO] Observation space:", env.observation_space)

        #### Train the model #######################################
        model = PPO("MlpPolicy",
                    env,
                    verbose=1
                    )
        model.learn(total_timesteps=100000) # 

        env = RLMAV()

        obs, info = env.reset(seed=7)

        for i in range(6*env.CTRL_FREQ):
            action, _states = model.predict(obs,
                                        deterministic=True
                                        )
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                obs = env.reset(seed=7)
        env.close()


if __name__ == "__main__":
    learn = learn_hover()
    learn.run()


