import numpy as np
from transformations import lla_to_mci

def generate_3d_trajectory(
    start_lat, start_lon, start_alt,
    v0_local=np.array([0.0, 1600.0, 0.0]),  # Local ENU velocity
    dt=1.0, T_max=7250, m=2000, theta0=0.0
):
    mu_moon = 4.9048695e12
    R_moon = 1.7374e6

    # Initial inertial position
    r0 = lla_to_mci(start_lat, start_lon, start_alt, t=0.0, theta0=theta0)

    # Define local ENU frame
    lat_rad = np.radians(start_lat)
    lon_rad = np.radians(start_lon)
    east = np.array([-np.sin(lon_rad), np.cos(lon_rad), 0])
    north = np.array([-np.sin(lat_rad)*np.cos(lon_rad), -np.sin(lat_rad)*np.sin(lon_rad), np.cos(lat_rad)])
    up = np.array([np.cos(lat_rad)*np.cos(lon_rad), np.cos(lat_rad)*np.sin(lon_rad), np.sin(lat_rad)])

    # Initial inertial velocity
    v0 = v0_local[0]*east + v0_local[1]*north + v0_local[2]*up

    # Setup simulation loop
    r = r0.copy()
    v = v0.copy()
    t = 0.0
    traj = []

    while True:
        r_mag = np.linalg.norm(r)
        h = r_mag - R_moon

        # === Thrust logic ===
        v_norm = np.linalg.norm(v)
        r_hat = r / np.linalg.norm(r)   # Local "up" direction
        h_safe = 100.0                  # Altitude threshold for terminal descent
        v_terminal = -2.0              # Target touchdown vertical velocity (m/s)
        v_descent = -10.0              # Controlled descent velocity (m/s)

        if v_norm > 50 and h > h_safe:
            # --- Phase 1: Braking burn to kill high velocity ---
            thrust_dir = -v / (v_norm + 1e-6)
            phase = 1

        elif h > h_safe:
            # --- Phase 2: Controlled descent toward -10 m/s along radial ---
            v_target = v_descent * r_hat
            v_error = v_target - v
            thrust_dir = v_error / (np.linalg.norm(v_error) + 1e-6)
            phase = 2

        elif h <= h_safe:
            # --- Phase 3: Controlled vertical descent using PD control ---
            v_radial = np.dot(v, r_hat)
            a_radial = np.dot(acc_gravity, r_hat)

            # PD controller: target v_terminal with damping
            v_error = v_terminal - v_radial
            Kp = 0.5
            Kd = 0.01
            a_cmd = Kp * v_error - Kd * v_radial - a_radial  # command acceleration upward

            # Thrust only in radial direction
            thrust_dir = r_hat
            thrust_mag = np.clip(m * a_cmd, 0, T_max)
            phase = 3

        # Gravity and thrust
        acc_gravity = -mu_moon * r / r_mag**3
        acc_desired = -acc_gravity + 0.5 * thrust_dir * (T_max / m)
        thrust_mag = np.clip(np.linalg.norm(acc_desired) * m, 0, T_max)
        acc_thrust = (thrust_mag / m) * thrust_dir
        acc = acc_gravity + acc_thrust

        # Integrate
        v += acc * dt
        r += v * dt
        t += dt

        traj.append([t, *r, *v, h, thrust_mag, *thrust_dir, phase])

        if h <= 100: #and np.linalg.norm(v) < 2.0:
            break

    return np.array(traj)