import numpy as np

class EKF_fusion:
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

    def measurement_model(self, x, u):
        """
        Inputs:
            x : 6x1 state vector (position + velocity)
            u : 3x1 control input (force vector in Newtons)

        Output:
            6x1 measurement prediction: [position; acceleration]
        """
        r = x[0:3].reshape((3,))
        r_norm = np.linalg.norm(r)

        # Gravitational acceleration
        a_gravity = -self.mu * r / (r_norm**3)

        # Control acceleration
        a_control = u / self.mass  # u is a force vector in N

        # Total predicted acceleration
        a_pred = a_control + a_gravity

        # Output: [position; acceleration]
        return np.vstack((r.reshape((3,1)), a_pred.reshape((3,1))))
    
    @staticmethod
    def dadr_dr(r, mu):
        """
        Compute the Jacobian of gravitational acceleration w.r.t. position.
        
        Inputs:
            r : 3-element position vector (numpy array)
            mu : gravitational parameter (scalar)

        Returns:
            3x3 Jacobian matrix
        """
        r = np.asarray(r)
        r_norm = np.linalg.norm(r)

        if r_norm < 1e-8:
            raise ValueError("Position norm too small; division by near-zero")

        I = np.eye(3)
        outer = np.outer(r, r)
        
        jacobian = -mu * ((I / r_norm**3) - (3 * outer) / r_norm**5)
        return jacobian

    def jacobian_C(self,x):
        r = x[0:3]
        C_top = np.hstack((np.eye(3), np.zeros((3,3))))
        C_bottom_left = self.dadr_dr(r, self.mu)
        C_bottom_right = np.zeros((3,3))
        C = np.vstack((C_top, np.hstack((C_bottom_left, C_bottom_right))))
        return C

    def predict(self, dt, u):
        # Propagate state (Euler)
        f = self.dynamics(self.x, u)
        self.x = self.x + dt * f

        # Jacobian of f at current state
        A = self.jacobian_A(self.x)
        Phi = np.eye(self.state_dim) + A * dt 

        # Predict covariance
        self.sigma = Phi @ self.sigma @ Phi.T + self.Q

    def update(self, y, u):
        """ Takes in a measurement vector y and updates the state and covariance. """
        # Jacobian of measurement model
        C = self.jacobian_C(self.x)

        # Measurement residual
        g = self.measurement_model(self.x,u)
        y_residual = y.reshape(-1, 1) - g

        # Kalman gain
        K = self.sigma @ C.T @ np.linalg.inv(C @ self.sigma @ C.T + self.R)

        # Update mean and covariance
        self.x = self.x + K @ y_residual
        self.sigma = self.sigma - K @ C @ self.sigma