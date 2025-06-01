from lunar_render import LunarRender
from crater_detector import CraterDetector
from visual_positioning import noise_multiplier
import numpy as np
import csv
import os

moon = LunarRender('WAC_ROI',debug=False)
moon.verbose = False
detector = CraterDetector()

csv_filename = 'crater_logs.csv'

# Write header if file doesn't exist
if not os.path.exists(csv_filename):
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'True_X', 'True_Y', 'True_Z',
            'ConfMult',
            'Crater1_U', 'Crater1_V','Crater1_Conf',
            'Crater2_U', 'Crater2_V','Crater2_Conf',
            'Crater3_U', 'Crater3_V','Crater3_Conf',
            'Crater4_U', 'Crater4_V','Crater4_Conf',
            'Crater5_U', 'Crater5_V','Crater5_Conf',
            'Crater6_U', 'Crater6_V','Crater6_Conf',
            'Crater7_U', 'Crater7_V','Crater7_Conf',
            'Crater8_U', 'Crater8_V','Crater8_Conf',
            'Crater9_U', 'Crater9_V','Crater9_Conf',
            'Crater10_U', 'Crater10_V','Crater10_Conf',
        ])

trajectory = np.genfromtxt('traj_fixed_LLA.csv',delimiter=',',skip_header=True)

for i, point in enumerate(trajectory):
    lat, lon, alt = point
    print(f"Processing Image #{i}")

    if alt <= 0:
        print("Altitude below zero, lander has landed")
        break

    tile = moon.render_ll(lat=lat, lon=lon, alt=alt, deg=True)
    detection = detector.detect_craters(tile)
    noise_mult = noise_multiplier(detection, 0.5)

    num_craters = min(detection.shape[0], 10)
    craters_in_img = []
    for crater in detection[:num_craters]:
        u, v, w, h, conf = crater
        craters_in_img.extend([u, v, conf])

    # padding = [0.0] * (3 * (10 - num_craters))

    # Flatten all values and convert to float for safety
    row = [float(lat), float(lon), float(alt), float(noise_mult)] + [float(x) for x in craters_in_img] #+ padding

    with open(csv_filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(row)
        csvfile.flush()
        os.fsync(csvfile.fileno())
        

print(f"\nProcessing complete!")
print(f"Data saved to {csv_filename}")
