from scipy import optimize
import csv
import numpy as np
from FinalProject.LunarLander.src.experiments.PNP_test import get_camera_position

training_data = np.genfromtxt('moon_training.csv',skip_header=1, delimiter=',')

uX = []
uY = []
vX = []
vY = []

def objective(opt_true_pos, verbose=False):
    f = 21e-3 # 21mm focal length
    pixel_size = 24e-6 # pixel size in meters
    focal_length = f/pixel_size # focal length in pixels
    # focal_length = 411.0
    # if isinstance(focal_length, np.ndarray):
    #     focal_length = focal_length.item()

    # print(f"Evaluating focal length: {focal_length:.2f} pixels")
    print(f"Evaluating global position: {opt_true_pos}")

    error = 0.0
    for i, point in enumerate(training_data):

        camera_matrix = np.array([
            [focal_length, 0, -256], 
            [0, focal_length, -256], 
            [0, 0, 1]
        ], dtype=np.float32)
        
        true_x, true_y = opt_true_pos[2*i:2*i+2]
        true_z = point[2]
        # true_x, true_y, true_z = point[:3]
        crater1 = point[3:8]
        crater2 = point[8:13]
        crater3 = point[13:18]

        image_points = np.array([
            [crater1[0], crater1[1]],
            [crater2[0], crater2[1]],
            [crater3[0], crater3[1]]
        ], dtype=np.float32)

        world_points = np.array([
            [crater1[2], crater1[3], crater1[4]],
            [crater2[2], crater2[3], crater2[4]],
            [crater3[2], crater3[3], crater3[4]]
        ], dtype=np.float32)

        dist_coeffs = np.zeros((4, 1))

        try:
            camera_pos, R, t = get_camera_position(
                world_points, 
                image_points, 
                camera_matrix, 
                dist_coeffs
            )
            uX.append(point[0])
            uY.append(camera_pos[0])
            vX.append(point[1])
            vY.append(camera_pos[1])

            if verbose:
                print(f"True Position: {point[0]}, {point[1]}, {point[2]}")
                print(f"Optimized True Position: {true_x}, {true_y}, {true_z}")
                print(f"Camera Position: {camera_pos}")
                print(f"Error: {true_x-camera_pos[0]:.2f}, {true_y-camera_pos[1]:.2f}, {true_z-camera_pos[2]:.2f}")
            error += np.linalg.norm(camera_pos - np.array([true_x, true_y, true_z])) ** 2
        except:
            error += 1e10


    return error


f = 21e-3 # 21mm focal length
pixel_size = 24e-6 # pixel size in meters
init_focal = f/pixel_size # focal length in pixels
init_focal = 411

init_true_pos = np.array([
    50000.0, 10000.0,
    30000.0, 15000.0,
    30000.0, -10000,
    10000.0, -10000.0,
], dtype=np.float32)

# Run optimization using BFGS
print("Starting focal length optimization...")

# bounds = [(100, 1500)]  # reasonable focal length range
options = {'eps': 1} # initial step size
result = optimize.minimize(objective, init_true_pos, method='L-BFGS-B', options=options) #, bounds=bounds)

print(f"Optimization completed!")
print(f"Optimal focal length: {result.x[0]:.2f} pixels")
print(f"Final error: {result.fun:.2e}")
print(f"Success: {result.success}")
print(f"Number of iterations: {result.nit}")

# Convert back to physical focal length
# optimal_focal_mm = result.x[0] * 24e-6 * 1000  # Convert to mm
# print(f"Optimal focal length: {optimal_focal_mm:.2f} mm")

print(f"\nFinal Objective:")
objective(result.x, verbose=True)

from scipy import stats

slope_u, intercept_u, r_value_u, p_value_u, std_err_u = stats.linregress(uX, uY)
print(f"\nU-coordinate relationship (true_x -> estimated_x):")
print(f"  Equation: estimated_x = {slope_u:.6f} * true_x + {intercept_u:.2f}")
print(f"  R-squared: {r_value_u**2:.4f}")
print(f"  Correlation coefficient: {r_value_u:.4f}")
print(f"  P-value: {p_value_u:.2e}")
print(f"  Standard error: {std_err_u:.4f}")

slope_v, intercept_v, r_value_v, p_value_v, std_err_v = stats.linregress(vX, vY)
print(f"\nV-coordinate relationship (true_y -> estimated_y):")
print(f"  Equation: estimated_y = {slope_v:.6f} * true_y + {intercept_v:.2f}")
print(f"  R-squared: {r_value_v**2:.4f}")
print(f"  Correlation coefficient: {r_value_v:.4f}")
print(f"  P-value: {p_value_v:.2e}")
print(f"  Standard error: {std_err_v:.4f}")