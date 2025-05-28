import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from transformations import lla2mcmf, mcmf2lla, convert_2d_trajectory_to_3d
from trajectory_generation import generate_trajectory

def plot_moon_global_view_with_trajectory(
    traj_3d, landing_vec,
    lat_step=30, lon_step=30,
    start_LLA=(5, 10, 100), target_LLA=(-20, 45, 0)
):
    R_moon = 1737.4
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])

    # Moon surface
    u, v = np.linspace(0, 2 * np.pi, 100), np.linspace(0, np.pi, 100)
    x = R_moon * np.outer(np.cos(u), np.sin(v))
    y = R_moon * np.outer(np.sin(u), np.sin(v))
    z = R_moon * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color='gray', alpha=0.6, edgecolor='none')

    # Grid lines
    for lon_deg in range(0, 360, lon_step):
        lon_rad = np.radians(lon_deg)
        lat_vals = np.linspace(-90, 90, 200)
        lat_rad = np.radians(lat_vals)
        x_line = R_moon * np.cos(lat_rad) * np.cos(lon_rad)
        y_line = R_moon * np.cos(lat_rad) * np.sin(lon_rad)
        z_line = R_moon * np.sin(lat_rad)
        ax.plot(x_line, y_line, z_line, color='black', linewidth=0.5)
        x_lbl = R_moon * np.cos(0) * np.cos(lon_rad)
        y_lbl = R_moon * np.cos(0) * np.sin(lon_rad)
        z_lbl = R_moon * np.sin(0)
        ax.text(x_lbl * 1.05, y_lbl * 1.05, z_lbl * 1.05, f"{lon_deg}°", color='black', fontsize=8)

    for lat_deg in range(-60, 90, lat_step):
        lat_rad = np.radians(lat_deg)
        lon_vals = np.linspace(0, 360, 200)
        lon_rad = np.radians(lon_vals)
        x_line = R_moon * np.cos(lat_rad) * np.cos(lon_rad)
        y_line = R_moon * np.cos(lat_rad) * np.sin(lon_rad)
        z_line = R_moon * np.ones_like(lon_rad) * np.sin(lat_rad)
        ax.plot(x_line, y_line, z_line, color='black', linewidth=0.5)
        x_lbl = R_moon * np.cos(lat_rad)
        y_lbl = 0
        z_lbl = R_moon * np.sin(lat_rad)
        ax.text(x_lbl * 1.05, y_lbl, z_lbl * 1.05, f"{lat_deg}°", color='blue', fontsize=8)

    # Plot trajectory and points
    start_xyz = lla2mcmf(start_LLA[1], start_LLA[0], start_LLA[2])
    target_xyz = lla2mcmf(target_LLA[1], target_LLA[0], target_LLA[2])
    ax.scatter(*start_xyz, color='green', s=50, label='Start')
    ax.scatter(*target_xyz, color='red', s=50, label='Landing')
    ax.text(*start_xyz * 1.02, 'Start', color='green')
    ax.text(*target_xyz * 1.02, 'Landing', color='red')
    ax.plot(traj_3d[:, 0], traj_3d[:, 1], traj_3d[:, 2], color='orange', linewidth=2, label='Trajectory')

    ax.set_title("Global Moon View with Trajectory")
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    ax.legend()
    ax.grid(False)
    plt.tight_layout()
    plt.show()

def plot_zoomed_trajectory_view(traj_3d, start_LLA, target_LLA):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])

    # Plot trajectory
    ax.plot(traj_3d[:, 0], traj_3d[:, 1], traj_3d[:, 2], color='orange', linewidth=2, label='Trajectory')

    # Start and end points
    start_xyz = lla2mcmf(start_LLA[1], start_LLA[0], start_LLA[2])
    target_xyz = lla2mcmf(target_LLA[1], target_LLA[0], target_LLA[2])
    ax.scatter(*start_xyz, color='green', s=50)
    ax.scatter(*target_xyz, color='red', s=50)
    ax.text(*start_xyz * 1.01, 'Start', color='green')
    ax.text(*target_xyz * 1.01, 'Landing', color='red')

    # Set zoom limits
    buffer = 10
    min_xyz = np.min(traj_3d, axis=0) - buffer
    max_xyz = np.max(traj_3d, axis=0) + buffer
    ax.set_xlim(min_xyz[0], max_xyz[0])
    ax.set_ylim(min_xyz[1], max_xyz[1])
    ax.set_zlim(min_xyz[2], max_xyz[2])

    ax.set_title("Zoomed-in Trajectory View (No Surface)")
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    ax.grid(False)
    plt.tight_layout()
    plt.show()

def plot_descent_trajectory(traj):
    # Extract values
    t = traj[:, 0]
    h = traj[:, 2]
    vx = traj[:, 3]
    vy = traj[:, 4]
    v_mag = np.sqrt(vx**2 + vy**2)
    thrust = traj[:, 5]
    phase = traj[:, 7]

    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(15, 6))

    # Altitude vs Time
    axs[0].plot(t, h)
    axs[0].set_xlabel('Time [s]')
    axs[0].set_ylabel('Altitude [m]')
    axs[0].set_title('Altitude vs Time')
    axs[0].grid(True)

    # Velocity profiles
    axs[1].plot(t, v_mag, label='|v|')
    axs[1].plot(t, vx, '--', label='vx')
    axs[1].plot(t, vy, '--', label='vy')
    axs[1].set_xlabel('Time [s]')
    axs[1].set_ylabel('Velocity [m/s]')
    axs[1].set_title('Velocity Profiles')
    axs[1].legend()
    axs[1].grid(True)

    # Thrust vs Time with phases
    axs[2].plot(t, thrust, label='Thrust [N]')
    axs[2].fill_between(t, 0, thrust, where=(phase == 1), alpha=0.2, label='Phase 1')
    axs[2].fill_between(t, 0, thrust, where=(phase == 2), alpha=0.2, label='Phase 2')
    axs[2].fill_between(t, 0, thrust, where=(phase == 3), alpha=0.2, label='Phase 3')
    axs[2].set_xlabel('Time [s]')
    axs[2].set_ylabel('Thrust [N]')
    axs[2].set_title('Thrust vs Time (by Phase)')
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()


traj = generate_trajectory()
plot_descent_trajectory(traj)

trajectory_2d = traj[:, [1, 2]]/1e3  # extract x (horizontal) and h (altitude) in km

#print("2D Trajectory:", trajectory_2d)

landing_lat = 10  # degrees
landing_lon = 270.0  # degrees

traj_3d, landing_vec = convert_2d_trajectory_to_3d(trajectory_2d, landing_lat, landing_lon)

#print("Landing Vector:", landing_vec)
#print("3D Trajectory:", traj_3d)

#print(traj_3d[0])
start_LLA = mcmf2lla(traj_3d[0])
#print("Start LLA:", start_LLA)

# Call with existing trajectory
plot_moon_global_view_with_trajectory(traj_3d, landing_vec, start_LLA=start_LLA, target_LLA=(landing_lat, landing_lon, 0))
plot_zoomed_trajectory_view(traj_3d, start_LLA, (landing_lat, landing_lon, 0))