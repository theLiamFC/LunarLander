import numpy as np

class LunarSimulator():
    def __init__(self,
                    true_state0, # true initial state of lander [x,y,z,vx,vy,vz] (m)
                    mu_state0, # initial guess state of lander [x,y,z,vx,vy,vz] (m)
                    cov0, # initial covariance of lander state
                    target, # target landing site [x,y,z=0] (m)
                    dt=0.1, # delta time for simulation (s)
                    ):
        pass
    def control(self,state):
        pass
    def noiseless_dynamics_step(self,state,input):
        pass
    def noisy_dynamics_step(self,state,input):
        pass
    def noiseless_measurement_step(self,state):
        pass
    def noisy_measurement_step(self,state):
        pass
    def simulate(self,state0,num_steps,seed=273,noisy=True):
        pass