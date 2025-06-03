import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from transformations import lla2mcmf, mcmf2lla, convert_traj_to_moon_fixed
from trajectory_generation import generate_3d_trajectory

def plot_moon_global_view_with_trajectory(
    traj_fixed_m,  # trajectory in Moon-fixed frame, in meters
    start_LLA, landing_LLA,
    lat_step=30, lon_step=30,
):
    R_moon = 1737.4  # [km]
    traj_km = traj_fixed_m / 1000.0  # Convert to kilometers

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])

    # Moon surface
    u, v = np.linspace(0, 2 * np.pi, 100), np.linspace(0, np.pi, 100)
    x = R_moon * np.outer(np.cos(u), np.sin(v))
    y = R_moon * np.outer(np.sin(u), np.sin(v))
    z = R_moon * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color='gray', alpha=0.6, edgecolor='none')

    # Latitude/longitude grid lines
    for lon_deg in range(0, 360, lon_step):
        lon_rad = np.radians(lon_deg)
        lat_vals = np.linspace(-90, 90, 200)
        lat_rad = np.radians(lat_vals)
        x_line = R_moon * np.cos(lat_rad) * np.cos(lon_rad)
        y_line = R_moon * np.cos(lat_rad) * np.sin(lon_rad)
        z_line = R_moon * np.sin(lat_rad)
        ax.plot(x_line, y_line, z_line, color='black', linewidth=0.5)
        ax.text(R_moon * 1.05 * np.cos(0) * np.cos(lon_rad),
                R_moon * 1.05 * np.cos(0) * np.sin(lon_rad),
                0,
                f"{lon_deg}°", color='black', fontsize=8)

    for lat_deg in range(-60, 90, lat_step):
        lat_rad = np.radians(lat_deg)
        lon_vals = np.linspace(0, 360, 200)
        lon_rad = np.radians(lon_vals)
        x_line = R_moon * np.cos(lat_rad) * np.cos(lon_rad)
        y_line = R_moon * np.cos(lat_rad) * np.sin(lon_rad)
        z_line = R_moon * np.sin(lat_rad) * np.ones_like(lon_rad)
        ax.plot(x_line, y_line, z_line, color='black', linewidth=0.5)
        ax.text(R_moon * 1.05 * np.cos(lat_rad), 0, R_moon * 1.05 * np.sin(lat_rad),
                f"{lat_deg}°", color='blue', fontsize=8)

    # Start and landing points
    start_xyz = lla2mcmf(start_LLA[1], start_LLA[0], start_LLA[2]) / 1000.0
    landing_xyz = lla2mcmf(landing_LLA[1], landing_LLA[0], landing_LLA[2]) / 1000.0
    ax.scatter(*start_xyz, color='green', s=50, label='Start')
    ax.scatter(*landing_xyz, color='red', s=50, label='Landing')
    ax.text(*start_xyz * 1.02, 'Start', color='green')
    ax.text(*landing_xyz * 1.02, 'Landing', color='red')

    # Trajectory
    ax.plot(traj_km[:, 0], traj_km[:, 1], traj_km[:, 2], color='orange', linewidth=2, label='Trajectory')

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
    # Extract data from updated trajectory format
    t = traj[:, 0]
    r = traj[:, 1:4]
    v = traj[:, 4:7]
    h = traj[:, 7]
    acc = traj[:, 8:11]
    thrust_mag_nominal = traj[:, 11]
    thrust_dir_nominal = traj[:, 12:15]
    thrust_mag_noisy = traj[:, 15]
    thrust_dir_noisy = traj[:, 16:19]
    phase = traj[:, 19]

    v_mag = np.linalg.norm(v, axis=1)

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Altitude vs Time
    axs[0].plot(t, h)
    axs[0].set_xlabel('Time [s]')
    axs[0].set_ylabel('Altitude [m]')
    axs[0].set_title('Altitude vs Time')
    axs[0].grid(True)

    # Velocity profiles
    axs[1].plot(t, v_mag, label='|v|')
    axs[1].plot(t, v[:, 0], '--', label='vx')
    axs[1].plot(t, v[:, 1], '--', label='vy')
    axs[1].plot(t, v[:, 2], '--', label='vz')
    axs[1].set_xlabel('Time [s]')
    axs[1].set_ylabel('Velocity [m/s]')
    axs[1].set_title('Velocity Components')
    axs[1].legend()
    axs[1].grid(True)

    # Thrust: Nominal vs Noisy
    axs[2].plot(t, thrust_mag_nominal, label='Nominal Thrust [N]', linewidth=2)
    axs[2].plot(t, thrust_mag_noisy, '--', label='Noisy Thrust [N]', linewidth=1.5)

    axs[2].fill_between(t, 0, thrust_mag_noisy, where=(phase == 1), alpha=0.15, label='Phase 1')
    axs[2].fill_between(t, 0, thrust_mag_noisy, where=(phase == 2), alpha=0.15, label='Phase 2')
    axs[2].fill_between(t, 0, thrust_mag_noisy, where=(phase == 3), alpha=0.15, label='Phase 3')

    axs[2].set_xlabel('Time [s]')
    axs[2].set_ylabel('Thrust [N]')
    axs[2].set_title('Thrust vs Time (Nominal and Noisy)')
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()


start_LLA = (0.0, 0.0, 100000)  # Latitude, Longitude, Altitude in meters

traj_inertial = generate_3d_trajectory(
    start_lat=start_LLA[0], start_lon=start_LLA[1], start_alt=start_LLA[2],  # 100 km above surface
    v0_local=np.array([1600, 0, 0])  # East/North/Up in m/s
)

traj_fixed = convert_traj_to_moon_fixed(traj_inertial)
landing_xyz = traj_fixed[-1, :3]   # Last position in Moon-fixed frame\

landing_lat,landing_lon,altitude_meters = mcmf2lla(landing_xyz)

landing_LLA = (landing_lat,landing_lon,altitude_meters)  # Target landing site in LLA (degrees, degrees, meters)
print(f"Landing LLA: {landing_LLA}")

# plot_moon_global_view_with_trajectory(
#     traj_fixed_m=traj_fixed,
#     start_LLA=start_LLA,
#     landing_LLA=landing_LLA
# )

plot_descent_trajectory(traj_inertial)