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
from plot_craters import get_crater_count

# TRAJECTORY INERTIAL INDEXING
TIME_IDX = 0
X_IDX = 1          # r[0] - Position X
Y_IDX = 2          # r[1] - Position Y  
Z_IDX = 3          # r[2] - Position Z
VX_IDX = 4         # v[0] - Velocity X
VY_IDX = 5         # v[1] - Velocity Y
VZ_IDX = 6         # v[2] - Velocity Z
ALTITUDE_IDX = 7   # h - Altitude
AX_IDX = 8         # acc[0] - Acceleration X
AY_IDX = 9         # acc[1] - Acceleration Y
AZ_IDX = 10        # acc[2] - Acceleration Z
THRUST_MAG_NOM_IDX = 11        # thrust_mag_nominal
THRUST_DIR_NOM_X_IDX = 12      # thrust_dir_nominal[0]
THRUST_DIR_NOM_Y_IDX = 13      # thrust_dir_nominal[1]
THRUST_DIR_NOM_Z_IDX = 14      # thrust_dir_nominal[2]
THRUST_MAG_NOISY_IDX = 15      # thrust_mag_noisy
THRUST_DIR_NOISY_X_IDX = 16    # thrust_dir_noisy[0]
THRUST_DIR_NOISY_Y_IDX = 17    # thrust_dir_noisy[1]
THRUST_DIR_NOISY_Z_IDX = 18    # thrust_dir_noisy[2]
PHASE_IDX = 19     # phase

if __name__ == "__main__":
    ############################################################
    # RUN TRAJECTORY
    ############################################################

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

    traj_all = np.hstack([traj_fixed_LLA, traj_inertial])

    print(traj_all.shape)

    # Create a DataFrame
    df_lla = pd.DataFrame(
        traj_all, 
        columns=[
            "Latitude_deg", "Longitude_deg", "Altitude_m",
            "Time",
            "X_Inertial", "Y_Inertial", "Z_Inertial",
            "VX_Inertial", "VY_Inertial", "VZ_Inertial",
            "Altitude_m",
            "X_Accel", "Y_Accel", "Z_Accel",
            "Thrust_Mag",
            "Thrust_Dir_x", "Thrust_Dir_Y", "Thrust_Dir_Z",
            "Thrust_Mag_N",
            "Thrust_Dir_x_N", "Thrust_Dir_Y_N", "Thrust_Dir_Z_N",
            "Phase"
        ]
    )

    # Save to CSV
    df_lla.to_csv("traj_all.csv", index=False)

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

    ############################################################
    # SIMULATION PARAMETERS
    ############################################################

    # CHOOSE VBN / IMU COMBINATION:
    SET_SIM = [True, True] # [VBN, IMU]             <==== EDIT THIS FOR TURNING OFF/ON VBN & IMU
    sim_mode_str = ""
    if SET_SIM[0] and SET_SIM[1]: sim_mode_str = "VBN & IMU"
    elif SET_SIM[0]: sim_mode_str = "VBN"
    elif SET_SIM[1]: sim_mode_str = "IMU"

    # CHOOSE TRAJECTORY NOISE LEVEL
    SET_NOISE = 5 # 1, 5                            <==== EDIT THIS FOR TRAJECTORY NOISE
    crater_log_fname = ""
    if SET_NOISE == 5:
        crater_log_fname = "crater_logs_noisy_01.csv"
    elif SET_NOISE == 1:
        crater_log_fname = "crater_logs_noisy_05.csv"
    crater_count = get_crater_count(crater_log_fname)

    ############################################################
    # RUN SIMULATION
    ############################################################

    # INITIAL STATE
    meas_dim = 6  # LLA measurement
    x0 = np.hstack((r0_inertial, v0_inertial)) # [x, y, z, vx, vy, vz]
    state_dim = x0.size
    sigma_std = 10000 * np.eye(state_dim)
    estimate_noise = np.linalg.cholesky(sigma_std) @ np.random.randn(6) 
    x0 = x0 + estimate_noise
    sigma0 = 1000 * sigma_std

    # INIT EKF FUSION
    mu = 4.9048695e12
    mass = 2000
    ekf = EKF_fusion(state_dim, meas_dim, mu, mass)
    ekf.set_initial_state(x0, sigma0)

    # PROCESS NOISE
    Q = 1e1*np.eye(state_dim)
    ekf.set_process_noise(Q)

    # MEASUREMENT NOISE
    if SET_SIM[0]: vbn_noise = 66**2 * np.eye(3) 
    else: vbn_noise = 1e20 * np.eye(3)
    if SET_SIM[1]: imu_noise = 1e-3 * np.eye(3)
    else: imu_noise = 1e20 * np.eye(3)
    R = np.block([
        [vbn_noise, np.zeros((3,3))],
        [np.zeros((3,3)),imu_noise]
    ])
    ekf.set_measurement_noise(R)

    # INIT LUNAR RENDER
    cam = Camera(r_mat=vbn_noise)
    cam.crater_log = crater_log_fname
    moon = LunarRender('WAC_ROI',debug=False)
    moon.verbose = False

    # INIT IMU
    from imu_simulator import IMUSimulator
    imu = IMUSimulator(mass=mass)

    # ALLOCATE EKF ESTIMATES HISTORY
    ekf_estimates = np.zeros((traj_fixed_LLA.shape[0], state_dim))
    measurements = np.zeros((traj_fixed_LLA.shape[0], meas_dim))
    ekf_estimates[0] = ekf.x.flatten()  # Store initial state estimate

    # ALLOCATE SIGMA HISTORY
    sigma_sqrt = np.zeros((traj_fixed_LLA.shape[0], state_dim))
    sigma_sqrt[0] = np.sqrt(np.diag(ekf.sigma))

    # TRIM DATA IF BELOW 0 (M) ALTITUDE
    traj_indexing = traj_fixed_LLA.copy()
    traj_fixed_LLA = traj_fixed_LLA[traj_indexing[:,2] >= 0]
    traj_inertial = traj_inertial[traj_indexing[:,2] >= 0]

    for i in range(1, traj_fixed_LLA.shape[0]):
        # TRUE STATE
        lat, lon, alt = traj_fixed_LLA[i,:]

        if alt <= 0.0: break

        # VISION BASED NAVIGATION MEASUREMENTS
        # tile = moon.render_ll(lat=lat,lon=lon,alt=alt,deg=True)
        if SET_SIM[0]:
            LLA_measure, mult = cam.get_position_global(i, alt, log=True, deg=True)
            lat_meas, lon_meas, alt_meas = LLA_measure
            R_new = np.block([
                [mult*vbn_noise, np.zeros((3,3))],
                [np.zeros((3,3)),imu_noise]
            ])        
            ekf.set_measurement_noise(R_new)
        else:
            lat_meas, lon_meas, alt_meas = np.zeros(3)

        # IMU MEASUREMENTS
        if SET_SIM[1]:
            thrust_mag_noisy = traj_inertial[i-1,THRUST_MAG_NOISY_IDX]
            thrust_dir_noisy = traj_inertial[i-1,THRUST_DIR_NOISY_X_IDX:THRUST_DIR_NOISY_Z_IDX+1]
            # a_mci_meas_imu = thrust_mag_noisy * thrust_dir_noisy + np.random.multivariate_normal(np.zeros(3),imu_noise) # simple noise
            a_mci_meas_imu = imu.get_acceleration(thrust_mag_noisy * thrust_dir_noisy) # biased noise
        else:
            a_mci_meas_imu = np.zeros(3)

        # COMBINED MEASUREMENTS 
        t = traj_inertial[i-1, 0]  # time in seconds
        r_mci_meas = lla_to_mci(lat_meas, lon_meas, alt_meas, t)
        full_meas = np.hstack((r_mci_meas, a_mci_meas_imu)) # Combine position and acceleration measurements
        measurements[i] = full_meas

        # NOMINAL THRUST INPUTS
        thrust_mag_nom = traj_inertial[i-1,THRUST_MAG_NOM_IDX]
        thrust_dir_nom = traj_inertial[i-1,THRUST_DIR_NOM_X_IDX:THRUST_DIR_NOM_Z_IDX+1]

        # EKF PREDICT & UPDATE
        ekf.predict(time_step, thrust_mag_nom * thrust_dir_nom)
        ekf.update(full_meas, thrust_mag_nom * thrust_dir_nom)

        # STORE EKF ESTIMATES
        ekf_estimates[i] = ekf.x.flatten()
        sigma_sqrt[i] = np.sqrt(np.diag(ekf.sigma))
        print(f"Sigma sqrt: {sigma_sqrt[i]}")

        print(f"Step {i}")
        print(f"True State (LLA): lat: {lat} deg, lon: {lon} deg, alt: {alt} m")
        print(f"Measurement (LLA): lat: {lat_meas} deg, lon: {lon_meas} deg, alt: {alt_meas} m")
        print(f"EKF Estimate (state): {ekf.x}")

    ############################################################
    # PLOTTING
    ############################################################

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
        end_time = len(time) - 5
        axs[i].plot(time[:end_time], true_pos[:end_time, i], label='True', color='black')
        axs[i].plot(time[:end_time], est_pos[:end_time, i], label='EKF Estimate', color='red', linestyle='--')

        # 1-sigma and 2-sigma bounds
        axs[i].fill_between(
            time[:end_time],
            est_pos[:end_time, i] - sigma_sqrt[:end_time, i],
            est_pos[:end_time, i] + sigma_sqrt[:end_time, i],
            color='orange', alpha=0.3, label='1σ'
        )
        axs[i].fill_between(
            time[:end_time],
            est_pos[:end_time, i] - 2*sigma_sqrt[:end_time, i],
            est_pos[:end_time, i] + 2*sigma_sqrt[:end_time, i],
            color='yellow', alpha=0.2, label='2σ'
        )
        axs[i].set_ylabel(labels[i])
        axs[i].legend()
        axs[i].grid(True)

    axs[2].set_xlabel('Time (s)')
    plt.suptitle(f'True vs EKF Estimated ECI Position Components ({sim_mode_str})')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Compute estimation error
    error = est_pos[:end_time] - true_pos[:end_time]  # shape: (N, 3)
    
    # Plot estimation error for each component
    fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    labels = ['x error (m)', 'y error (m)', 'z error (m)']
    
    for i in range(3):
        axs[i].plot(time[:end_time], error[:end_time, i], color='blue')
        axs[i].set_ylabel(labels[i])
        axs[i].grid(True)
    
    axs[2].set_xlabel('Time (s)')
    axs[3].plot(time[:end_time], crater_count[:end_time])
    axs[3].set_xlabel('Time (s)')
    axs[3].set_ylabel('Crater Count')
    axs[3].grid(True)
    
    plt.suptitle(f'EKF Estimation Error in ECI Position Components ({sim_mode_str})')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.show()