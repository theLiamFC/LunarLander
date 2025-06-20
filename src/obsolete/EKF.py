import numpy as np

class EKF:
    def __init__(self, state_dim, meas_dim, mu, mass=2000):
        """
        Initialize the EKF.
        state_dim: Dimension of the state vector.
        meas_dim: Dimension of the measurement vector.
        mu: Gravitational parameter.
        """
        self.state_dim = state_dim
        self.meas_dim = meas_dim
        self.mu = mu
        self.mass = mass

        self.x = np.zeros((state_dim, 1))
        self.sigma = np.eye(state_dim)
        self.Q = np.eye(state_dim)
        self.R = np.eye(meas_dim)

    def set_initial_state(self, x0, sigma0):
        self.x = x0.reshape(-1, 1)
        self.sigma = sigma0

    def set_process_noise(self, Q):
        self.Q = Q

    def set_measurement_noise(self, R):
        self.R = R

    def get_state(self):
        return self.x

    def get_covariance(self):
        return self.sigma
    
    def dynamics(self, x, u):
        r = x[0:3].flatten()
        v = x[3:6].flatten()
        mu = self.mu
        norm_r = np.linalg.norm(r)
        a = (-mu * r / norm_r**3) + (u / self.mass)  # u is thrust
        dxdt = np.hstack((v, a))
        return dxdt.reshape(-1, 1)

    def jacobian_A(self, x):
        I3 = np.eye(3)
        mu = self.mu

        r = x[0:3].flatten()
        norm_r = np.linalg.norm(r)        
        rrT = np.outer(r, r)
        da_dr = -mu * ((I3 / norm_r**3) - (3 * rrT) / norm_r**5)

        A = np.zeros((6, 6))
        A[0:3, 3:6] = I3
        A[3:6, 0:3] = da_dr
        return A

    def measurement_model(self, x):
        return x[0:3].reshape(-1, 1)

    def jacobian_C(self):
        C = np.zeros((3, 6))
        C[0:3, 0:3] = np.eye(3)
        
        return C #np.hstack([np.eye(2), np.zeros((2,4))])

    def predict(self, dt, u):
        # Propagate state (Euler)
        f = self.dynamics(self.x, u)
        self.x = self.x + dt * f

        # Jacobian of f at current state
        A = self.jacobian_A(self.x)
        Phi = np.eye(self.state_dim) + A * dt 

        # Predict covariance
        self.sigma = Phi @ self.sigma @ Phi.T + self.Q

    def update(self, y):
        """ Takes in a measurement vector y and updates the state and covariance. """
        # Jacobian of measurement model
        C = self.jacobian_C()

        # Measurement residual
        g = self.measurement_model(self.x)
        y_residual = y.reshape(-1, 1) - g

        # Kalman gain
        K = self.sigma @ C.T @ np.linalg.inv(C @ self.sigma @ C.T + self.R)

        # Update mean and covariance
        self.x = self.x + K @ y_residual
        self.sigma = self.sigma - K @ C @ self.sigma

    def update_R(self, R):
        self.R = R