from lunar_simulator import LunarSimulator
from camera import Camera
from trajectory_generation import generate_3d_trajectory
from transformations import convert_traj_to_moon_fixed, mcmf_traj_to_lla, lla_to_mci
import numpy as np
import pandas as pd
import EKF
from lunar_render import LunarRender


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
    thrust_total = np.linalg.norm(thrust_dir, axis=1) * thrust_mag  # Total thrust vector

    cam = Camera()
    moon = LunarRender('../WAC_ROI',debug=False)
    moon.verbose = False
    
    # set initial state
    x0 = np.hstack((r0_inertial, v0_inertial))  # [x, y, z, vx, vy, vz]
    state_dim = x0.size
    meas_dim = 3  # LLA measurement
    sigma0 = 10000 * np.eye(state_dim)

    # initialize EKF
    ekf = EKF(state_dim, meas_dim, mu_moon = 4.9048695e12)
    ekf.set_initial_state(x0, sigma0)
    ekf.set_process_noise(np.eye(state_dim))
    ekf.set_measurement_noise(np.eye(meas_dim))

    # Storage for EKF estimates
    ekf_estimates = np.zeros((traj_fixed_LLA.shape[0], state_dim))
    measurements = np.zeros((traj_fixed_LLA.shape[0], 3))
    
    for i in range(traj_fixed_LLA.shape[0]):
        lat, lon, alt = traj_fixed_LLA[i,:]
        tile = moon.render_ll(lat=lat,lon=lon,alt=alt,deg=True)
        measurement = cam.get_position_global_hack(tile, alt) # ouputs lat, lon, altitide (deg, deg, km) of camera position in world frame
        measurements[i] = measurement

        # get measurements
        lat_meas = measurement[0]
        lon_meas = measurement[1]
        alt_meas = measurement[2]

        # convert LLA measurements to inertial coordinates
        t = traj_inertial[i, 0]  # time in seconds
        r_mci_meas = lla_to_mci(lat_meas, lon_meas, alt_meas, t)

        # EKF predict and update
        ekf.predict(time_step,thrust_total[i])
        ekf.update(r_mci_meas)

        # Store EKF estimate
        ekf_estimates[i] = ekf.x.flatten()

        print(f"Step {i}")
        print(f"True State (LLA): lat: {lat} deg, lon: {lon} deg, alt: {alt} m")
        print(f"Measurement (LLA): lat: {measurement[0]} deg, lon: {measurement[1]} deg, alt: {measurement[2]} m")
        print(f"EKF Estimate (state): {ekf.x}")
    
    measurements
    
    # lunar_sim = LunarSimulator(
    #     target,  # target landing site [x,y,z,vx=0,vy=0,vz=0] (m)
    #     true_state0,  # true initial state of lander [x,y,z,vx,vy,vz] (m)
    #     mu_state0,  # initial guess state of lander [x,y,z,vx,vy,vz] (m)
    #     cov0,  # initial covariance of lander state
    #     q_mat,  # process noise covariance matrix
    #     r_mat,  # measurement noise covariance matrix
    #     runtime=100,  # duration of simulation (s)
    #     dt=0.1,  # delta time for simulation (s)
    #     LROC_folder="WAC_ROI", # local folder containing LROC images
    #     fov=45 # simulated fov of camera in degrees
    # )

    # LunarSimulator.simulate(
    #     state0, 
    #     seed=273, 
    #     noisy=True
    # )

    # LunarSimulator.plot()

