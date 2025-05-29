import numpy as np
from lunar_render import LunarRender
from crater_detector import CraterDetector
#from src.camera import VisualOdometry
from imu_simulator import IMUSimulator

class LunarSimulator:
    def __init__(
        self,
        target,  # target landing site [x,y,z,vx=0,vy=0,vz=0] (m)
        true_state0,  # true initial state of lander [x,y,z,vx,vy,vz] (m)
        mu_state0,  # initial guess state of lander [x,y,z,vx,vy,vz] (m)
        cov0,  # initial covariance of lander state
        q_mat,  # process noise covariance matrix
        r_mat,  # measurement noise covariance matrix
        runtime=100,  # duration of simulation (s)
        dt=0.1,  # delta time for simulation (s)
        LROC_folder="WAC_ROI", # local folder containing LROC images
        fov=45 # simulated fov of camera in degrees
    ):
        self.num_steps = runtime / dt + 1

        self.target = target

        self.state_dim = true_state0.shape[0]
        self.true_state = np.zeros((self.state_dim, self.num_steps))
        self.true_state[:, 0] = true_state0
        self.mu_state = np.zeros((self.state_dim, self.num_steps))
        self.mu_state[:, 0] = mu_state0
        self.cov = np.zeros((cov0.shape[0], cov0.shape[1], self.num_steps))
        self.cov[:, :, 0] = cov0

        self.q_mat = q_mat
        self.r_mat = r_mat

        self.lunar_render = LunarRender(LROC_folder, fov=fov)
        self.tiles = np.empty((2,1), dtype=object) # [time, Tile]

        self.crater_detector = CraterDetector()
        self.visual_odometer = VisualOdometry()

        self.imu_simulator = IMUSimulator()

    def simulate(self, state0, seed=273, noisy=True):
        pass

    def plot(self):
        # plot run
        pass

    def _control(self, state):
        pass

    def _noiseless_dynamics_step(self, state, input):
        pass

    def _noisy_dynamics_step(self, state, input):
        return self._noiseless_dynamics_step(state,input) + np.random.multivariate_normal(np.zeros(self.q_mat.shape[0]),self.q_mat)

    def _noiseless_measurement_step(self, state):
        # return concatenated private get functions
        pass

    def _noisy_measurement_step(self, state):
        return self._noiseless_measurement_step(state) + np.random.multivariate_normal(np.zeros(self.r_mat.shape[0]),self.r_mat)

    def _get_imu(self,state,control):
        # not sure if our control is force or acceleration => might need update
        self.imu_simulator.get_acceleration(state,control)
        pass

    def _get_radar(self):
        # simulate radar from current state
        pass

    def _get_rel_vel(self, last_tile, curr_tile):
        return self.visual_odometer.get_odometry(last_tile, curr_tile)

    def _get_glob_pos(self):
        # call crater detection function on last image
        # return global position function on crater locations
        pass

    def _ekf_step(self):
        pass
