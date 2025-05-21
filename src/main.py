from lunar_simulator import LunarSimulator

if __name__ == "__main__":
    lunar_sim = LunarSimulator(
        target,  # target landing site [x,y,z,vx=0,vy=0,vz=0] (m)
        true_state0,  # true initial state of lander [x,y,z,vx,vy,vz] (m)
        mu_state0,  # initial guess state of lander [x,y,z,vx,vy,vz] (m)
        cov0,  # initial covariance of lander state
        q_mat,  # process noise covariance matrix
        r_mat,  # measurement noise covariance matrix
        runtime=100,  # duration of simulation (s)
        dt=0.1,  # delta time for simulation (s)
        LROC_folder="WAC_ROI", # local folder containing LROC images
        fov=45 # simulated fov of camera in degrees
    )

    LunarSimulator.simulate(
        state0, 
        seed=273, 
        noisy=True
    )

    LunarSimulator.plot()

