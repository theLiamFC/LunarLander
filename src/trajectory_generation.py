import numpy as np
from transformations import lla_to_mci

def generate_3d_trajectory(
    start_lat, start_lon, start_alt,
    v0_local=np.array([0.0, 1600.0, 0.0]),
    dt=1.0, T_max=7000, m=2000, theta0=0.0,
    sigma_mag=0.01, sigma_dir=0.001, random_seed=273
):
    np.random.seed(random_seed)
    
    mu_moon = 4.9048695e12
    R_moon = 1.7374e6

    r0 = lla_to_mci(start_lat, start_lon, start_alt, t=0.0, theta0=theta0)

    lat_rad = np.radians(start_lat)
    lon_rad = np.radians(start_lon)
    east = np.array([-np.sin(lon_rad), np.cos(lon_rad), 0])
    north = np.array([-np.sin(lat_rad)*np.cos(lon_rad), -np.sin(lat_rad)*np.sin(lon_rad), np.cos(lat_rad)])
    up = np.array([np.cos(lat_rad)*np.cos(lon_rad), np.cos(lat_rad)*np.sin(lon_rad), np.sin(lat_rad)])

    v0 = v0_local[0]*east + v0_local[1]*north + v0_local[2]*up

    r = r0.copy()
    v = v0.copy()
    t = 0.0
    traj = []

    while True:
        r_mag = np.linalg.norm(r)
        h = r_mag - R_moon

        v_norm = np.linalg.norm(v)
        r_hat = r / (r_mag + 1e-6)
        h_safe = 100.0
        v_terminal = -2.0
        v_descent = -10.0

        if v_norm > 50 and h > h_safe:
            thrust_dir_cmd = -v / (v_norm + 1e-6)
            phase = 1
        elif h > h_safe:
            v_target = v_descent * r_hat
            v_error = v_target - v
            thrust_dir_cmd = v_error / (np.linalg.norm(v_error) + 1e-6)
            phase = 2

        else:
            v_radial = np.dot(v, r_hat)
            v_error = v_terminal - v_radial
            thrust_dir_cmd = v_error * r_hat - v
            thrust_dir_cmd /= np.linalg.norm(thrust_dir_cmd) + 1e-6
            phase = 3

        # No smoothing (direct assignment)
        thrust_dir_nominal = thrust_dir_cmd

        # Gravity and thrust
        acc_gravity = -mu_moon * r / (r_mag**3)
        acc_desired = -acc_gravity + 0.5 * thrust_dir_nominal * (T_max / m)
        thrust_mag_nominal = np.clip(np.linalg.norm(acc_desired) * m, 0, T_max)

        # Noisy thrust
        mag_scale_noise = 1.0 + np.random.normal(0.0, sigma_mag)
        thrust_mag_noisy = thrust_mag_nominal * mag_scale_noise

        dir_perturb = np.random.normal(0.0, sigma_dir, size=3)
        thrust_dir_noisy = thrust_dir_nominal + dir_perturb
        thrust_dir_noisy /= np.linalg.norm(thrust_dir_noisy) + 1e-6

        acc_thrust = (thrust_mag_noisy / m) * thrust_dir_noisy
        acc = acc_gravity + acc_thrust

        # Integrate
        v += acc * dt
        r += v * dt
        t += dt

        traj.append([
            t, *r, *v, h, *acc,
            thrust_mag_nominal, *thrust_dir_nominal,
            thrust_mag_noisy, *thrust_dir_noisy,
            phase
        ])

        if h <= 1:
            break

    return np.array(traj)
