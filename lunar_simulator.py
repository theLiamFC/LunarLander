import numpy as np
from LunarRender import LunarRender

class LunarSimulator:
    def __init__(
        self,
        target,  # target landing site [x,y,z,vx=0,vy=0,vz=0] (m)
        true_state0,  # true initial state of lander [x,y,z,vx,vy,vz] (m)
        mu_state0,  # initial guess state of lander [x,y,z,vx,vy,vz] (m)
        cov0,  # initial covariance of lander state
        q_mat,  # process noise covariance matrix
        r_mat,  # measurement noise covariance matrix
        runtime,  # duration of simulation (s)
        dt=0.1,  # delta time for simulation (s)
        LROC_folder="WAC_ROI", # local folder containing LROC images
        fov=45 # simulated fov of camera in degrees
    ):
        self.num_steps = runtime / dt + 1

        self.target = target

        self.true_state = np.zeros((true_state0.shape[0], self.num_steps))
        self.true_state[:, 0] = true_state0
        self.mu_state = np.zeros((mu_state0.shape[0], self.num_steps))
        self.mu_state[:, 0] = mu_state0
        self.cov = np.zeros((cov0.shape[0], cov0.shape[1], self.num_steps))
        self.cov[:, :, 0] = cov0

        self.q_mat = q_mat
        self.r_mat = r_mat

        self.lunar_render = LunarRender('WAC_ROI', fov=fov)
        self.tiles = np.empty((2,1), dtype=object) # [time, Tile]

        # initiate crater detection class
        # initiate relative velocity class

    def control(self, state):
        pass

    def noiseless_dynamics_step(self, state, input):
        pass

    def noisy_dynamics_step(self, state, input):
        pass

    def noiseless_measurement_step(self, state):
        # return concatenated private get functions
        pass

    def noisy_measurement_step(self, state):
        # return noiseless measurement + noise
        pass

    def simulate(self, state0, num_steps, seed=273, noisy=True):
        pass

    def __get_imu__(self):
        # simulate imu from current state
        pass

    def __get_radar__(self):
        # simulate radar from current state
        pass

    def __get_rel_vel__(self):
        # call relative velocity function on last two images
        pass

    def __get_glob_pos__(self):
        # call crater detection function on last image
        # return global position function on crater locations
        pass
