import numpy as np
import matplotlib.pyplot as plt
from lunar_render import LunarRender, Tile
from visual_odometer import VisualOdometer

def generate_quadratic_trajectory(num_points=100):
    """Generate a quadratic trajectory in 3D space"""
    # X coordinates: linear progression
    time = np.linspace(0, 100, num_points)

    x_coords = 1 * time**2
    y_coords = -4 * time**2
    y_coords = 0 * time
    z_coords = 100000 + 5000 * np.sin(np.linspace(0, 2*np.pi, num_points))

    x_vel = 2*1*time
    y_vel = 2*-4*time
    y_vel = 0*time
    z_vel = 5000 * np.cos(np.linspace(0, 2*np.pi, num_points))
    
    # Create time stamps
    times = np.linspace(0, 19, num_points)
    
    # Combine into state vectors [x, y, z]
    states = np.zeros((num_points, 3))
    true_vel = np.zeros((num_points, 3))

    for i in range(num_points):
        states[i,:] = [x_coords[i], y_coords[i], z_coords[i]]
        true_vel[i,:] = [x_vel[i], y_vel[i], z_vel[i]]

    print(true_vel)
    
    return states, true_vel, times

def main():
    """Main test function"""
    print("Generating quadratic trajectory...")
    states, true_vel, times = generate_quadratic_trajectory(200)
    
    # Use mock render since rasterio might not be available
    print("Initializing LunarRender class")
    lunar_render = LunarRender('src/WAC_ROI', foc=21, size=512)
    
    # Use mock odometer since OpenCV might not be available
    print("Initializing VisualOdometer class")
    visual_odometer = VisualOdometer()

    
    print("Generating lunar surface images...")
    tiles = []
    for i, state in enumerate(states):
        x, y, z = state
        tile = lunar_render.render_m(x, y, z, time=times[i])
        tiles.append(tile)
    
    print("Estimating velocities using visual odometry...")
    estimated_velocities_x = []
    estimated_velocities_y = []
    
    plotting = np.random.randint(0, len(states) - 2)
    for i in range(len(states) - 1):
        # Estimate velocities using visual odometry
        plot = False
        if i == plotting:
            plot = True
        else:
            plot = False
        est_vx, est_vy = visual_odometer.get_velocity(
            states[i], states[i+1], tiles[i], tiles[i+1], plot
        )
        
        estimated_velocities_x.append(est_vx)
        estimated_velocities_y.append(est_vy)
    
    print("Plotting results...")
    # Create the comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    time_points = times[:-1]  # One less point since we're calculating differences
    
    # X velocity comparison
    ax1.plot(time_points, true_vel[:-1,0], 'b-', label='True Velocity X', linewidth=2)
    ax1.plot(time_points, estimated_velocities_x, 'r--', label='Estimated Velocity X', linewidth=2)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Velocity X (m/s)')
    ax1.set_title('X Velocity Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Y velocity comparison
    ax2.plot(time_points, true_vel[:-1,1], 'b-', label='True Velocity Y', linewidth=2)
    ax2.plot(time_points, estimated_velocities_y, 'r--', label='Estimated Velocity Y', linewidth=2)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Velocity Y (m/s)')
    ax2.set_title('Y Velocity Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print some statistics
    print("\nVelocity Statistics:")
    print(f"True X velocity range: {min(true_vel[:,0]):.2f} to {max(true_vel[:,0]):.2f} m/s")
    print(f"True Y velocity range: {min(true_vel[:,1]):.2f} to {max(true_vel[:,1]):.2f} m/s")
    print(f"Estimated X velocity range: {min(estimated_velocities_x):.2f} to {max(estimated_velocities_x):.2f} m/s")
    print(f"Estimated Y velocity range: {min(estimated_velocities_y):.2f} to {max(estimated_velocities_y):.2f} m/s")
    
    # Calculate RMS error
    rms_error_x = np.sqrt(np.mean([(t - e)**2 for t, e in zip(true_vel[:,0], estimated_velocities_x)]))
    rms_error_y = np.sqrt(np.mean([(t - e)**2 for t, e in zip(true_vel[:,1], estimated_velocities_y)]))
    
    print(f"\nRMS Error X: {rms_error_x:.2f} m/s")
    print(f"RMS Error Y: {rms_error_y:.2f} m/s")

if __name__ == "__main__":
    main()
