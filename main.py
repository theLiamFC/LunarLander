from src.visual_positioning import Camera
from src.trajectory_generation import generate_3d_trajectory
from src.transformations import convert_traj_to_moon_fixed, mcmf_traj_to_lla, lla_to_mci, mci_to_lla, get_enu_to_mci_rotation
import numpy as np
import pandas as pd
from src.EKF_fusion import EKF_fusion
from src.lunar_render import LunarRender
import matplotlib.pyplot as plt
from src.crater_detector import get_crater_count

MOON_RADIUS_M = 1_737_400

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

    traj_all = np.hstack([traj_fixed_LLA, traj_inertial])

    TIME_STEPS = 600
    traj_fixed_LLA = traj_fixed_LLA[:TIME_STEPS]
    traj_all = traj_all[:TIME_STEPS]
    traj_inertial = traj_inertial[:TIME_STEPS]

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
    df_lla.to_csv("src/csv_files/traj_all.csv", index=False)

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
    SET_SIM = [True, True, True] # [VBN, IMU, TRN]             <==== EDIT THIS FOR TURNING OFF/ON VBN & IMU
    sim_mode_str = ""
    if SET_SIM[0] and SET_SIM[1]: sim_mode_str = "VBN & IMU"
    elif SET_SIM[0]: sim_mode_str = "VBN"
    elif SET_SIM[1]: sim_mode_str = "IMU"

    # CHOOSE TRAJECTORY NOISE LEVEL
    SET_NOISE = 5 # 1, 5                            <==== EDIT THIS FOR TRAJECTORY NOISE
    crater_log_fname = ""
    if SET_NOISE == 5:
        crater_log_fname = "src/csv_files/crater_logs_noisy_01.csv"
    elif SET_NOISE == 1:
        crater_log_fname = "src/csv_files/crater_logs_noisy_05.csv"
    crater_count = get_crater_count(crater_log_fname)

    ############################################################
    # RUN SIMULATION
    ############################################################

    # INITIAL STATE
    meas_dim = 9  # LLA measurement
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
    if SET_SIM[2]: 
        # trn_noise = 1e2 * np.eye(3)
        trn_noise = np.array(
            [[5e3,0,0],
            [0,5e3,0],
            [0,0,1e10]]
        )
    else: trn_noise = 1e20 * np.eye(3)
    R = np.block([
        [vbn_noise, np.zeros((3,3)), np.zeros((3,3))],
        [np.zeros((3,3)), trn_noise, np.zeros((3,3))],
        [np.zeros((3,3)), np.zeros((3,3)), imu_noise]
    ])
    ekf.set_measurement_noise(R)

    # INIT LUNAR RENDER
    cam = Camera(r_mat=vbn_noise)
    cam.crater_log = crater_log_fname
    moon = LunarRender('src/WAC_ROI',debug=False)
    moon.verbose = False

    # INIT IMU
    from src.imu_simulator import IMUSimulator
    imu = IMUSimulator(mass=mass)

    # INIT ODOMETER
    from src.visual_odometer import VisualOdometer
    trn = VisualOdometer()

    # ALLOCATE EKF ESTIMATES HISTORY
    ekf_estimates = np.zeros((traj_fixed_LLA.shape[0], state_dim))
    measurements = np.zeros((traj_fixed_LLA.shape[0], meas_dim))
    ekf_estimates[0] = ekf.x.flatten()  # Store initial state estimate

    # ALLOCATE SIGMA HISTORY
    sigma_sqrt = np.zeros((traj_fixed_LLA.shape[0], state_dim))
    sigma_sqrt[0] = np.sqrt(np.diag(ekf.sigma))

    # ALLOCATE TILE ARRAY
    tile_history = []

    # TRIM DATA IF BELOW 0 (M) ALTITUDE
    traj_indexing = traj_fixed_LLA.copy()
    traj_fixed_LLA = traj_fixed_LLA[traj_indexing[:,2] >= 0]
    traj_inertial = traj_inertial[traj_indexing[:,2] >= 0]

    rolling_average = np.zeros((traj_fixed_LLA.shape[0],3))

    for i in range(1, traj_fixed_LLA.shape[0]):
        # TRUE STATE
        lat, lon, alt = traj_fixed_LLA[i,:]

        if alt <= 0.0: break

        # VISION BASED NAVIGATION MEASUREMENTS
        tile = moon.render_ll(lat=lat,lon=lon,alt=alt,deg=True, time=i)
        if SET_SIM[0]:
            LLA_measure, mult = cam.get_position_global(i, alt, log=True, deg=True)
            lat_meas, lon_meas, alt_meas = LLA_measure

            new_vbn_noise = mult*vbn_noise

            R_new = np.block([
                [mult*vbn_noise, np.zeros((3,3)), np.zeros((3,3))],
                [np.zeros((3,3)), trn_noise, np.zeros((3,3))],
                [np.zeros((3,3)), np.zeros((3,3)), imu_noise]
            ])     
            ekf.set_measurement_noise(R_new)
        else:
            new_vbn_noise = vbn_noise
            lat_meas, lon_meas, alt_meas = np.zeros(3)

        # TERRAIN RELATIVE NAVIGATION MEASUREMENTS
        if SET_SIM[2]:
            if i == 1:
                v_mci_meas = ekf.x[3:6].flatten()
                v_mci_meas_r = v_mci_meas
                new_trn_noise = 1e20 * trn_noise
            else:
                try:
                    t1, t2 = tile_history[-1].time, tile.time
                    
                    lat1, lon1, alt1 = mci_to_lla(measurements[i-1,0],measurements[i-1,1],measurements[i-1,2])
                    vx, vy = trn.get_velocity(alt1, alt_meas, tile_history[-1], tile, plot=False)
                    print(f"Linear Velocities: vx={vx}, vy={vy}")
                    v_alt_meas = (traj_fixed_LLA[i,2] - traj_fixed_LLA[i-1,2]) / (t2 - t1)

                    enu_velocity = np.array([vx, vy, v_alt_meas])  # ENU velocities in m/s
                    rotation = get_enu_to_mci_rotation(np.radians(lat), np.radians(lon))
                    v_mci_meas = rotation @ enu_velocity
                    v_mci_meas_r = v_mci_meas

                    if i > 5:
                        sliding_window = np.vstack([v_mci_meas,measurements[i-5:i,3:6]])
                        v_mci_meas_r = np.mean(sliding_window,axis=0)
                    rolling_average[i,:] = v_mci_meas_r

                    if np.linalg.norm(v_mci_meas-measurements[i-1,3:6]) > 1000:
                        new_trn_noise = 1e20 * trn_noise
                    else:
                        new_trn_noise = trn_noise

                    print(f"Measured V: {v_mci_meas}")
                    print(f"True V: {traj_inertial[i, VX_IDX:VZ_IDX+1]}")
                except: # not enough feature matches
                    v_mci_meas = ekf.x[3:6].flatten()
                    v_mci_meas_r = v_mci_meas
                    new_trn_noise = 1e20 * trn_noise

            tile_history.append(tile)
        else:
            v_mci_meas = ekf.x[3:6].flatten()
            v_mci_meas_r = v_mci_meas

            new_trn_noise = 1e20 * trn_noise

        R_new = np.block([
            [new_vbn_noise, np.zeros((3,3)), np.zeros((3,3))],
            [np.zeros((3,3)), new_trn_noise, np.zeros((3,3))],
            [np.zeros((3,3)), np.zeros((3,3)), imu_noise]
        ])     
        ekf.set_measurement_noise(R_new)

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
        full_meas = np.hstack((r_mci_meas, v_mci_meas_r, a_mci_meas_imu)) # Combine position and acceleration measurements
        measurements[i] = np.hstack((r_mci_meas, v_mci_meas, a_mci_meas_imu))

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
        # print(f"EKF Estimate (state): {ekf.x}")

    ############################################################
    # PLOTTING
    ############################################################


    # Extract time and true ECI positions from traj_inertial
    time = traj_inertial[:, 0]
    true_pos = traj_inertial[:, 1:4]  # columns 1,2,3 are x,y,z in ECI
    end_time = len(time) - 5

    # Extract estimated ECI positions from EKF
    est_pos = ekf_estimates[:, 0:3]  # columns 0,1,2 are x,y,z

    figv, axsv = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ['x (m)', 'y (m)', 'z (m)']

    for i in range(3):
        axsv[i].plot(time[1:end_time], traj_inertial[1:end_time, VX_IDX+i], label='True Vel', color='black')
        axsv[i].plot(time[1:end_time], measurements[1:end_time, 3+i], label='Est Vel', color='red')
        axsv[i].plot(time[1:end_time], rolling_average[1:end_time, i], label='Avg Vel', color='blue')

    # Plot
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ['x (m)', 'y (m)', 'z (m)']

    for i in range(3):
        # Plot only up to 100 seconds
        axs[i].plot(time[1:end_time], true_pos[1:end_time, i], label='True', color='black')
        axs[i].plot(time[1:end_time], est_pos[1:end_time, i], label='EKF Estimate', color='red', linestyle='--')

        # 1-sigma and 2-sigma bounds
        axs[i].fill_between(
            time[1:end_time],
            est_pos[1:end_time, i] - sigma_sqrt[1:end_time, i],
            est_pos[1:end_time, i] + sigma_sqrt[1:end_time, i],
            color='orange', alpha=0.3, label='1σ'
        )
        axs[i].fill_between(
            time[1:end_time],
            est_pos[1:end_time, i] - 2*sigma_sqrt[1:end_time, i],
            est_pos[1:end_time, i] + 2*sigma_sqrt[1:end_time, i],
            color='yellow', alpha=0.2, label='2σ'
        )
        axs[i].set_ylabel(labels[i])
        axs[i].legend()
        axs[i].grid(True)

        # Add y-axis scaling calculation
        y_min = min(np.min(true_pos[1:end_time, i]), np.min(est_pos[1:end_time, i]))
        y_max = max(np.max(true_pos[1:end_time, i]), np.max(est_pos[1:end_time, i]))
        y_range = y_max - y_min
        padding = 0.1 * y_range  # 10% padding
        axs[i].set_ylim(y_min - padding, y_max + padding)

    axs[2].set_xlabel('Time (s)')
    plt.suptitle(f'True vs EKF Estimated ECI Position Components ({sim_mode_str})')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Compute estimation error
    error = est_pos[1:end_time] - true_pos[1:end_time]  # shape: (N, 3)
    
    # Plot estimation error for each component
    if not SET_SIM[0]: num_sub_plot = 3
    else: num_sub_plot = 4
    fig, axs = plt.subplots(num_sub_plot, 1, figsize=(10, 8), sharex=True)
    labels = ['x error (m)', 'y error (m)', 'z error (m)']
    
    for i in range(3):
        axs[i].plot(time[1:end_time], error[:, i], color='blue')
        axs[i].set_ylabel(labels[i])
        axs[i].grid(True)
        if i == 0:
            axs[i].set_title('EKF Error in X Position')
        if i == 1:
            axs[i].set_title('EKF Error in Y Position')
        if i == 2:
            axs[i].set_title('EKF Error in Z Position')
    
    axs[2].set_xlabel('Time (s)')
    if SET_SIM[0]:
        axs[3].plot(time[1:end_time], crater_count[1:end_time])
        axs[3].set_xlabel('Time (s)')
        axs[3].set_ylabel('Crater Count')
        axs[3].grid(True)
        axs[3].set_title('Crater Count vs Time')
    
    plt.suptitle(f'EKF Estimation Error in ECI Position Components ({sim_mode_str})')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    print(f"Min ERROR: {np.min(np.abs(error[1:end_time,:]),axis=0)}")
    print(f"MAX ERROR: {np.max(np.abs(error[1:end_time,:]),axis=0)}")
    print(f"Average ERROR: {np.mean(np.abs(error[1:end_time,:]), axis=0)}")

plt.show()