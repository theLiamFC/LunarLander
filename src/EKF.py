import numpy as np

class EKF:
    def __init__(self, state_dim, meas_dim):
        """
        Initialize the EKF.
        :param state_dim: Dimension of the state vector.
        :param meas_dim: Dimension of the measurement vector.
        """
        # dimensions
        self.state_dim = state_dim
        self.meas_dim = meas_dim

        # current state and covariance
        self.x = np.zeros((state_dim, 1))  
        self.sigma = np.eye(state_dim)         
        self.Q = np.eye(state_dim)        
        self.R = np.eye(meas_dim)         

    def set_initial_state(self, x0, P0):
        """
        Set the initial state and covariance.
        """
        self.x = x0
        self.sigma = P0

    def set_process_noise(self, Q):
        """
        Set the process noise covariance.
        """
        self.Q = Q

    def set_measurement_noise(self, R):
        """
        Set the measurement noise covariance.
        """
        self.R = R

    def predict(self, u, f, F_jacobian):
        """
        EKF prediction step.
        :param u: Control input
        :param f: Nonlinear state transition function, f(x, u)
        :param F_jacobian: Function to compute Jacobian of f w.r.t x, F(x, u)
        """
        # Predict state
        self.x = f(self.x, u)
        # Predict covariance
        F = F_jacobian(self.x, u)
        self.sigma = F @ self.sigma @ F.T + self.Q

    def update(self, z, h, H_jacobian):
        """
        EKF update step.
        :param z: Measurement
        :param h: Nonlinear measurement function, h(x)
        :param H_jacobian: Function to compute Jacobian of h w.r.t x, H(x)
        """
        # Measurement prediction
        z_pred = h(self.x)
        # Innovation
        y = z - z_pred
        # Measurement Jacobian
        H = H_jacobian(self.x)
        # Innovation covariance
        S = H @ self.sigma @ H.T + self.R
        # Kalman gain
        K = self.sigma @ H.T @ np.linalg.inv(S)
        # Update state
        self.x = self.x + K @ y
        # Update covariance
        I = np.eye(self.state_dim)
        self.sigma = (I - K @ H) @ self.sigma

    def get_state(self):
        """
        Get the current state estimate.
        """
        return self.x

    def get_covariance(self):
        """
        Get the current covariance estimate.
        """
        return self.sigma