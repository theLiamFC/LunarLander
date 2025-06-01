from lunar_render import LunarRender, locate_crater
from crater_detector import CraterDetector
from FinalProject.LunarLander.src.experiments.PNP_test import get_camera_position
import numpy as np
import csv
import os

moon = LunarRender('WAC_ROI',debug=False)
moon.verbose = False
detector = CraterDetector()

minU = -25000.0
maxU  = 80000.0
minV = -10000.0
maxV = 10000.0
alt = 80000.0

np.random.seed(273)  # For reproducibility
u_values = np.random.uniform(minU, maxU, 100)
v_values = np.random.uniform(minV, maxV, 100)
alt_values = np.full(100, alt)

test_points = np.vstack((u_values, v_values, alt_values)).T.astype(np.float32)

csv_filename = 'moon_training.csv'

# Write header if file doesn't exist
if not os.path.exists(csv_filename):
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'True_X', 'True_Y', 'True_Z', 
            'Crater1_U', 'Crater1_V', 'Crater1_GU', 'Crater1_GV', 'Crater1_Z',
            'Crater2_U', 'Crater2_V', 'Crater2_GU', 'Crater2_GV', 'Crater2_Z',
            'Crater3_U', 'Crater3_V', 'Crater3_GU', 'Crater3_GV', 'Crater3_Z'
        ])

successful_points = 0
total_points = len(test_points)

for i, point in enumerate(test_points):
    print(f"Processing point {i+1}/{total_points}: [{point[0]:.1f}, {point[1]:.1f}, {point[2]:.1f}]")
    
    try:
        tile = moon.render(u=point[0], v=point[1], alt=point[2])

        detection = detector.detect_craters(tile)

        num_craters = detection.shape[0]
        if num_craters < 3:
            print(f"  Skipped: Only {num_craters} craters detected (need 3+)")
            continue

        craters_in_img = []
        for j, crater in enumerate(detection[:3,:]):
            u, v, w, h, conf = crater

            gu, gv = locate_crater(tile, u, v)

            craters_in_img.extend([u, v, gu, gv, 0.0])

        row = list(point) + craters_in_img
        
        # Append to CSV file immediately
        with open(csv_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row)
        
        successful_points += 1
        print(f"  Success: Row {successful_points} written to CSV")
        
    except Exception as e:
        print(f"  Error: {str(e)}")
        continue

print(f"\nProcessing complete!")
print(f"Successfully processed: {successful_points}/{total_points} points")
print(f"Data saved to {csv_filename}")
