import cv2
import numpy as np
import pandas as pd
from src.crater_detector import CraterDetector
from src.transformations import lla2mcmf, mcmf2lla
from src.lunar_render import LunarRender, pixel_to_lat_lon, locate_crater
import time
"""

NOTES:
1. Camera object should initialize a crater_detector object inside it --> will be needed to calculate translational information 
2. Methods still need to be tested. Basic stuff is in there and the steps are written out with steps and sources.
3. Camera matrix needs to be created/talked about

"""

MOON_RADIUS_M = 1_737_400 # radius of moon in meters

class Camera():
    def __init__(self, r_mat, K=None):
        self.r_mat = r_mat / 100
        self.meas_dim = self.r_mat.shape[0]
        self.orb = cv2.ORB_create()
        self.detector = CraterDetector()
        self.crater_log = "src/csv_files/crater_logs_noisy_05.csv"
        self.df = pd.read_csv(self.crater_log, header=0)
        self.craters = self.df.values
        
    def noise_multiplier(self, predictions, thresh=0.5):
        num_craters = predictions.shape[0]        
        mult = 100 #for when number of valid craters detected > 0 
        
        if num_craters > 0:
            num_valid = 0
            running_total = 0
            for crater in predictions:
                if crater[-1] > thresh:
                    num_valid+=1
                    running_total += 1 - crater[-1] #exp(1-p_1)*exp(1-p_2)*...
            if num_valid > 0:
                mult = running_total / num_valid**2
        return mult
    
    def noise_multiplier_log(self, img_num):
        return self.craters[img_num, 2]

    def crater_noise(self, predictions, thresh = 0.5):
        mult = self.noise_multiplier(predictions, thresh) 
        return np.random.multivariate_normal(np.zeros(self.meas_dim), mult*self.r_mat).reshape(self.meas_dim,1)
            
    def get_position_global(self, tile, alt, log=True, deg=True, scale=7):    
        #tile is either the image generated tile or its the tile index number from the trajectory list
        
        if log:
            lat = np.float64(self.craters[tile, 0])
            lon = np.float64(self.craters[tile, 1])
            alt = np.float64(self.craters[tile, 2])
            mult = self.craters[tile,3]
            mult = 1 + np.float64(mult)

            print(f"MULT IS: {mult}")
            
            # Uncomment for debugging
            # if tile > 800 and mult != 100:
            #     print(f"Testing: {lat,lon,alt}")
            #     moon = LunarRender("WAC_ROI")
            #     tile = moon.render_ll(lat=lat, lon=lon, alt=alt, deg=True)
            #     self.detector.view_craters(tile)

            if mult == 101:
                mult = 5
                lat, lon, alt = np.array([lat, lon, alt]) + np.random.multivariate_normal(np.zeros(3), mult * self.r_mat / MOON_RADIUS_M)
            else:
                lat, lon, alt = np.array([lat, lon, alt]) + np.random.multivariate_normal(np.zeros(3), mult * self.r_mat / MOON_RADIUS_M)

        else:
            predictions = self.detector.detect_craters(tile)
            mult = self.noise_multiplier(predictions, 0.5)
            gu, gv = np.array([tile.u, tile.v]).reshape(2,1) + self.crater_noise(predictions)
            lat, lon = pixel_to_lat_lon(gu, gv, deg=True)
                
        print(f"Vals: {lat, lon, alt}")
        return np.array([lat, lon, alt]).flatten(), mult
    
def noise_multiplier(predictions, thresh=0.5):
    print(f"predictions; {predictions}")
    num_craters = predictions.shape[0]
    print(f"Num craters: {num_craters}")
    mult = 100 
        
    if num_craters > 0:
        num_valid = 0
        running_total = 0
        for crater in predictions:
            if crater[-1] > thresh:
                num_valid+=1
                running_total += 1 - crater[-1]
        if num_valid > 0:
            mult = running_total / num_valid
    return mult
    
        
    
        
if __name__ == "__main__":
    cam = Camera()
    K = cam.get_K()
    print(cam.get_K()) #returns the camera intrinsic matrix
    
    from lunar_render import LunarRender
    from crater_detector import CraterDetector
    
    moon = LunarRender('../WAC_ROI',debug=False)
    # tile = moon.render(u=0, v=0, alt=50000)
    tile = moon.render_ll(lat=0,lon=-30,alt=50000,deg=True)
    
    detector = CraterDetector()
    detector.view_craters(tile)
    print(cam.get_position_global(tile)) # ouputs lat, lon, altitide (deg, deg, km) of camera position in world frame

        
        
        