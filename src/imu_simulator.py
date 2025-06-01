import numpy as np
import matplotlib.pyplot as plt
import random

class IMUSimulator:
    def __init__(self, mass=2.0, altitude=0.0):
        self.mass = mass # might not need if we are outputting control as acceleration not force
        self.radius_moon = 1738000  # mean radius of the Moon in meters
        self.gravity_surface = 1.625  # gravity on the surface of the Moon in m/s²
        
        # Error parameters
        self.bias = [0.01, -0.008, 0.015]  # Bias for x, y, z (m/s²)
        self.noise_std = 0.2  # Base noise level
        self.bias_instability_std = 0.01  # Standard deviation for bias instability noise
        self.bias_instability = np.array([0.0, 0.0, 0.0])  # Initial bias instability
        self.saturation_limit = 3.0  # Saturation limit for acceleration (m/s²)

    def gravity(self, alt):
        """
        Scales gravity based on altitutde

        Parameters
        ----------
        alt : float
            Meters of altitude.
        
        Returns
        ----------
        g : float
            Gravitational acceleration
        """
        r = self.radius_moon + alt
        g = self.gravity_surface * (self.radius_moon / r) ** 2
        return g

    def update_bias_instability(self):
        """
        Randomly updates bias.
        
        Returns
        ----------
        bias_instability : float
            Std of bias noise
        """
        noise = np.random.normal(0, self.bias_instability_std, 3)
        self.bias_instability += noise
        return self.bias_instability
    
    def apply_saturation(self, acc):
        """
        Saturates acceleration measurements at high values.

        Parameters
        ----------
        acc : np.array
            Acceleration vector.
        
        Returns
        ----------
        acc : np.array
            Saturated acceleration vector.
        """
        acc_sat = []
        for a in acc:
            if a > self.saturation_limit:
                acc_sat.append(self.saturation_limit)
            elif a < -self.saturation_limit:
                acc_sat.append(-self.saturation_limit)
            else:
                acc_sat.append(a)
        return acc_sat

    def get_acceleration(self, state, force):
        """
        Simulates IMU measured acceleration given position and control force.

        Parameters
        ----------
        state : np.array
            State vector.
        force : np.array
            Force vector.
        
        Returns
        ----------
        final_acc : np.array
            Acceleration vector with error / noise.
        """
        alt = state[0]
        gravity_vector = [0, 0, -self.gravity(alt)]
        ideal_acc = [f / self.mass + g for f, g in zip(force, gravity_vector)]
        
        # 3. Update and apply bias instability
        bias_inst = self.update_bias_instability()
        
        # 4. Apply bias, bias instability, and noise
        acc_with_bias = [
            a + b + bi + random.gauss(0, self.noise_std) 
            for a, b, bi in zip(ideal_acc, self.bias, bias_inst)
        ]
        
        # 6. Apply saturation
        final_acc = self.apply_saturation(acc_with_bias)
        
        return final_acc

# Parameters
if __name__ == "__main__":
    mass = 2.0
    amplitude = 10.0  # Force amplitude in Newtons
    frequency = 1.0  # Frequency in Hz
    sampling_rate = 200  # Samples per second
    duration = 10.0  # Duration in seconds
    altitude = 10000  # Altitude in meters (10 km above lunar surface)

    # Time array
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

    # Force input: sine wave on X, zero on Y and Z
    force_x = amplitude * np.sin(2 * np.pi * frequency * t)
    force_y = np.zeros_like(t)
    force_z = np.zeros_like(t)

    # Create IMU instance
    imu = IMUSimulator(mass=mass, altitude=altitude)

    # Get acceleration data
    acceleration = [imu.get_acceleration([altitude],[fx, fy, fz]) for fx, fy, fz in zip(force_x, force_y, force_z)]

    # Convert to numpy array for easier plotting
    acceleration = np.array(acceleration)

    # Plotting
    plt.figure(figsize=(10, 7.5))
    plt.plot(t, acceleration[:, 0], label='IMU Acceleration X', color='#00b8c4')
    plt.plot(t, acceleration[:, 1], label='IMU Acceleration Y', color='#ffb07c')
    plt.plot(t, acceleration[:, 2], label='IMU Acceleration Z', color='#8b0000')
    plt.plot(t, force_x / mass, label='True Acceleration X', color='#00b8c4', linestyle='dashed')
    plt.plot(t, force_y / mass, label='True Acceleration Y', color='#ffb07c', linestyle='dashed')
    plt.plot(t, force_z / mass, label='True Acceleration Z', color='#8b0000', linestyle='dashed')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Acceleration (m/s^2)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
