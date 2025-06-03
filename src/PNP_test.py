import numpy as np
import cv2
import csv
from lunar_render import LunarRender, locate_crater
from crater_detector import CraterDetector

def get_camera_position(object_points, image_points, camera_matrix, dist_coeffs=None, method=cv2.SOLVEPNP_SQPNP):
    """
    Calculate camera position in world coordinates using PnP solver.
    
    Parameters:
    - object_points: 3D points in world coordinates (Nx3 array)
    - image_points: 2D points in image coordinates (Nx2 array)
    - camera_matrix: Camera intrinsic matrix (3x3)
    - dist_coeffs: Distortion coefficients (optional, defaults to no distortion)
    - method: PnP solver method (default: SOLVEPNP_ITERATIVE)
    
    Returns:
    - camera_position: 3D position of camera in world coordinates
    - rotation_matrix: 3x3 rotation matrix
    - translation_vector: 3x1 translation vector
    """
    if dist_coeffs is None:
        dist_coeffs = np.zeros((4, 1))

    object_points = np.array(object_points, dtype=np.float32).reshape(-1, 3)
    image_points = np.array(image_points, dtype=np.float32).reshape(-1, 2)
    
    success, rvec, tvec = cv2.solvePnP(
        object_points, 
        image_points, 
        camera_matrix, 
        dist_coeffs, 
        flags=method
    )
    
    if not success:
        raise Exception("solvePnP failed to find a solution")
    
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    
    camera_position = -rotation_matrix.T @ tvec

    camera_position[2] *= 100 # convert z axis from pixels to meters (100m / px)
    
    return camera_position.flatten(), rotation_matrix, tvec.flatten()

# Example usage
if __name__ == "__main__":

    moon = LunarRender('WAC_ROI',debug=False)
    moon.verbose = False
    detector = CraterDetector()

    f = 21e-3 # 21mm focal length
    pixel_size = 24e-6 # pixel size in meters
    focal = f/pixel_size # focal length in pixels
    focal = 411 # optimized focal length in pixels
    
    camera_matrix = np.array([
        [focal, 0, -256], 
        [0, focal, -256], 
        [0, 0, 1]
    ], dtype=np.float32)

    test_points = np.array([
        [50000.0, 10000.0, 80000.0], # px, px, m
        [30000.0, 15000.0, 80000.0],
        [30000.0, -10000, 80000.0],
        [10000.0, -10000.0, 80000.0]
    ], dtype=np.float32)

    csv_data = []
    csv_data.append(['True_X', 'True_Y', 'True_Z', 'Crater1_U', 'Crater1_V', 'Crater2_U', 'Crater2_V', 'Crater3_U', 'Crater3_V'])

    for point in test_points:
        
        print("-"*50)
        print(f"True Position:\n{point}")
        tile = moon.render(u=point[0], v=point[1], alt=point[2])

        detection = detector.detect_craters(tile)

        num_craters = detection.shape[0]
        if num_craters < 3:
            print("Not enough points detected for PnP.")
            continue

        image_points = np.zeros((num_craters, 2), dtype=np.float32)
        world_points = np.zeros((num_craters, 3), dtype=np.float32)

        top_3_craters = []
        for i, crater in enumerate(detection[:3,:]):
            u, v, w, h, conf = crater

            gu, gv = locate_crater(tile, u, v)

            top_3_craters.extend([u, v, gu, gv, 0.0])

            image_points[i,:] = np.array([u, v], dtype=np.float32)
            world_points[i,:] = np.array([gu, gv, 0.0], dtype=np.float32)

        row = list(point) + top_3_craters
        csv_data.append(row)

        csv_filename = 'moon_training.csv'
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(csv_data)

        print(f"\nData saved to {csv_filename}")

        dist_coeffs = np.zeros((4, 1))

        try:
            camera_pos, R, t = get_camera_position(
                world_points, 
                image_points, 
                camera_matrix, 
                dist_coeffs
            )

            print("Estimated Position:\n", camera_pos)
            print("Translation Vector:", t)
            print("Rotation Matrix:\n", R)
            print(f"Error:{camera_pos - point}\n")
        except Exception as e:
            print(f"Error: {e}")

    
    # try:
    #     # Method 1: Using iterative solver (works with 3+ points)
    #     camera_pos, R, t = get_camera_position(
    #         world_points, 
    #         image_points, 
    #         camera_matrix, 
    #         dist_coeffs
    #     )
        
    #     print("Camera Position (Iterative):", camera_pos)
    #     print("Translation Vector:", t)
    #     print("Rotation Matrix:\n", R)
        
    #     # # Method 2: Using P3P solver (requires exactly 4 points)
    #     # solutions = get_camera_position_p3p(
    #     #     world_points, 
    #     #     image_points, 
    #     #     camera_matrix, 
    #     #     dist_coeffs
    #     # )
        
    #     # print(f"\nP3P found {len(solutions)} solution(s):")
    #     # for i, sol in enumerate(solutions):
    #     #     print(f"Solution {i+1} - Camera Position:", sol['camera_position'])
            
    # except Exception as e:
    #     print(f"Error: {e}")
