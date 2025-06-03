from lunar_simulator import LunarSimulator
from visual_positioning import Camera
from trajectory_generation import generate_3d_trajectory
from transformations import convert_traj_to_moon_fixed, mcmf_traj_to_lla, lla_to_mci
import numpy as np
import pandas as pd
from EKF import EKF
from EKF_fusion import EKF_fusion
from lunar_render import LunarRender
import matplotlib.pyplot as plt


if __name__ == "__main__":
    start_LLA = (0.0, 0.0, 100000)  # Latitude, Longitude, Altitude in meters

    # traj.append([t, *r, *v, h, thrust_mag, *thrust_dir, phase])
    traj_inertial = generate_3d_trajectory(
        start_lat=start_LLA[0], start_lon=start_LLA[1], start_alt=start_LLA[2],  # 100 km above surface
        v0_local=np.array([1600, 0, 0])  # East/North/Up in m/s
    )
    traj_fixed = convert_traj_to_moon_fixed(traj_inertial)

    # Lat (deg), long (deg), altitude (meters)
    traj_fixed_LLA = mcmf_traj_to_lla(traj_fixed)
    #print(traj_fixed_LLA)
 
    # Create a DataFrame
    df_lla = pd.DataFrame(traj_fixed_LLA, columns=["Latitude_deg", "Longitude_deg", "Altitude_m"])

    # Save to CSV
    df_lla.to_csv("traj_fixed_LLA.csv", index=False)

    # Assuming traj is a NumPy array with shape (N, 13+)
    initial_row = traj_inertial[0]

    # Extract position and velocity from columns 1:4 and 4:7
    r0_inertial = initial_row[1:4]  # x, y, z
    v0_inertial = initial_row[4:7]  # vx, vy, vz
    time_step = traj_inertial[1, 0] - traj_inertial[0, 0]

    # Extract control inputs
    thrust_mag = traj_inertial[:,8] # Thrust magnitude
    thrust_dir = traj_inertial[:,9:12] # Thrust direction (unit vector)
    thrust_total = thrust_dir * thrust_mag[:, np.newaxis]  # Total thrust vector

    
    # set initial state
    meas_dim = 6  # LLA measurement
    x0 = np.hstack((r0_inertial, v0_inertial)) # [x, y, z, vx, vy, vz]
    state_dim = x0.size
    sigma_std = 1000 * np.eye(state_dim)
    estimate_noise = np.linalg.cholesky(sigma_std) @ np.random.randn(6) 

    # Add noise to the initial state
    x0 = x0 + estimate_noise 
    sigma0 = 1000 * sigma_std

    # # initialize EKF
    # ekf = EKF(state_dim, meas_dim, mu = 4.9048695e12)
    # ekf.set_initial_state(x0, sigma0)
    # Q = 1*np.eye(state_dim)
    # ekf.set_process_noise(Q)
    # R = 1000 * np.eye(meas_dim) 
    # ekf.set_measurement_noise(R)

    # # Storage for EKF estimates
    # ekf_estimates = np.zeros((traj_fixed_LLA.shape[0], state_dim))
    # measurements = np.zeros((traj_fixed_LLA.shape[0], 3))
    # ekf_estimates[0] = ekf.x.flatten()  # Store initial state estimate

    # # Store sqrt of diagonal of covariance (sigma) for each time step
    # sigma_sqrt = np.zeros((traj_fixed_LLA.shape[0], state_dim))
    # sigma_sqrt[0] = np.sqrt(np.diag(ekf.sigma))

    # initialize EKF Fusion
    mu = 4.9048695e12
    mass = 2000
    ekf = EKF_fusion(state_dim, meas_dim, mu, mass)
    ekf.set_initial_state(x0, sigma0)
    Q = 1*np.eye(state_dim)
    ekf.set_process_noise(Q)
    R = 1e3 * np.eye(meas_dim) 
    ekf.set_measurement_noise(R)

    cam = Camera(r_mat=R/100)
    moon = LunarRender('WAC_ROI',debug=False)

    moon.verbose = False

    # Storage for EKF estimates
    ekf_estimates = np.zeros((traj_fixed_LLA.shape[0], state_dim))
    measurements = np.zeros((traj_fixed_LLA.shape[0], meas_dim))
    ekf_estimates[0] = ekf.x.flatten()  # Store initial state estimate

    # Store sqrt of diagonal of covariance (sigma) for each time step
    sigma_sqrt = np.zeros((traj_fixed_LLA.shape[0], state_dim))
    sigma_sqrt[0] = np.sqrt(np.diag(ekf.sigma))

    for i in range(1, traj_fixed_LLA.shape[0]):
        lat, lon, alt = traj_fixed_LLA[i,:]

        tile = moon.render_ll(lat=lat,lon=lon,alt=alt,deg=True)
        # lat_meas,lon_meas,alt_meas  = cam.get_position_global_hack(tile, alt) # ouputs lat, lon, altitide (deg, deg, km) of camera position in world frame
        LLA_measure, mult = cam.get_position_global(i, alt, log=True, deg=True)

        lat_meas, lon_meas, alt_meas = LLA_measure

        ekf.set_measurement_noise(R*10*mult)

        u = thrust_total[i - 1]
        r = traj_inertial[i][1:4].flatten()
        v = traj_inertial[i][4:7].flatten()
        norm_r = np.linalg.norm(r)
        a = (-mu * r / norm_r**3) + (u / mass)  # u is thrust
        print(f"ACCEL: {a}")

        print(f"NOISE: {np.random.multivariate_normal(np.zeros(6),R)[3:7]}")
        a_mci_meas = a + np.random.multivariate_normal(np.zeros(6),R)[3:7]
        # TODO: MAKE SURE THIS t IS CORRECT, MIGHT NEED TO OFFSET by i - 1 also...
        t = traj_inertial[i-1, 0]  # time in seconds
        r_mci_meas = lla_to_mci(lat_meas, lon_meas, alt_meas, t)
        # a_mci_meas = traj_inertial  # FILL IN IMU MEASUREMENT HERE (3,1) acceleration in MCI frame

        full_meas = np.hstack((r_mci_meas, a_mci_meas)) # Combine position and acceleration measurements
        measurements[i] = full_meas

        # convert LLA measurements to inertial coordinates

        # EKF predict and update
        ekf.predict(time_step,thrust_total[i - 1])
        ekf.update(full_meas,thrust_total[i - 1])

        # Store EKF estimate
        ekf_estimates[i] = ekf.x.flatten()
        sigma_sqrt[i] = np.sqrt(np.diag(ekf.sigma))

        print(f"Step {i}")
        print(f"True State (LLA): lat: {lat} deg, lon: {lon} deg, alt: {alt} m")
        print(f"Measurement (LLA): lat: {lat_meas} deg, lon: {lon_meas} deg, alt: {alt_meas} m")
        print(f"EKF Estimate (state): {ekf.x}")

    # Extract time and true ECI positions from traj_inertial
    time = traj_inertial[:, 0]
    true_pos = traj_inertial[:, 1:4]  # columns 1,2,3 are x,y,z in ECI

    # Extract estimated ECI positions from EKF
    est_pos = ekf_estimates[:, 0:3]  # columns 0,1,2 are x,y,z

    # Plot
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ['x (m)', 'y (m)', 'z (m)']

    for i in range(3):
        # Plot only up to 100 seconds
        idx_100s = np.where(time <= 1400)[0]
        axs[i].plot(time[idx_100s], true_pos[idx_100s, i], label='True', color='black')
        axs[i].plot(time[idx_100s], est_pos[idx_100s, i], label='EKF Estimate', color='red', linestyle='--')

        # 1-sigma and 2-sigma bounds
        axs[i].fill_between(
            time,
            est_pos[:, i] - sigma_sqrt[:, i],
            est_pos[:, i] + sigma_sqrt[:, i],
            color='orange', alpha=0.3, label='1σ'
        )
        axs[i].fill_between(
            time,
            est_pos[:, i] - 2*sigma_sqrt[:, i],
            est_pos[:, i] + 2*sigma_sqrt[:, i],
            color='yellow', alpha=0.2, label='2σ'
        )
        axs[i].set_ylabel(labels[i])
        axs[i].legend()
        axs[i].grid(True)

    axs[2].set_xlabel('Time (s)')
    plt.suptitle('True vs EKF Estimated ECI Position Components')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Compute estimation error
    error = est_pos - true_pos  # shape: (N, 3)
    
    # Plot estimation error for each component
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ['x error (m)', 'y error (m)', 'z error (m)']
    
    for i in range(3):
        axs[i].plot(time, error[:, i], color='blue')
        axs[i].set_ylabel(labels[i])
        axs[i].grid(True)
    
    axs[2].set_xlabel('Time (s)')
    plt.suptitle('EKF Estimation Error in ECI Position Components')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.show()