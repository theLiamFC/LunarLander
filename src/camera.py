import cv2
import numpy as np
from crater_detector import CraterDetector
from transformations import lla2mcmf, mcmf2lla
from lunar_render import pixel_to_lat_lon
"""

NOTES:
1. Camera object should initialize a crater_detector object inside it --> will be needed to calculate translational information 
2. Methods still need to be tested. Basic stuff is in there and the steps are written out with steps and sources.
3. Camera matrix needs to be created/talked about



"""

class Camera():
    def __init__(self, K=None):
        self.orb = cv2.ORB_create()
        self.detector = CraterDetector()
        
        if K is None:
            f = 21e-3 #23mm focal length
            pixel_size = 24e-6
            focal = f/pixel_size
            self.K = np.array([[focal, 0,  -256], 
                                [0, focal, -256], 
                                [0, 0, 1]])
            self.focal = focal
        else:
            self.K = K #K represents the camera intrinsic matrix --> this needs to be input --> input as a 3x3
        
    def get_K(self):
        return self.K
    
    def get_matches(self,img1,img2, num_matches = 20, plot=False): #currently using ORB but can switch to SIFT (might be worth testing)
        #https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html
        #https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html
        #keypoints are tuples of (x,y)
        kp1, des1 = self.orb.detectAndCompute(img1, None)
        kp2, des2 = self.orb.detectAndCompute(img2, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        matches = matches[:num_matches] 
    
        #code lines copied from https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html
        points1 = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
        points2 = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
        
        if plot:
            return kp1, kp2
        else: #default
            return points1, points2, matches #returns them in the proper numpy format

    #just for visualization purposes
    def plot_matches(self,img1,img2, N=20):
        kp1, kp2, matches = self.get_matches(img1,img2, num_matches=N, plot=True)
        img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:N], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    def calc_essential(self, img1, img2):
        p1, p2, matches = self.get_matches(img1,img2) #first get corresponding points
        E = cv2.findEssentialMat(p1, p2, self.K)
        return E, p1, p2
    
    def recover_pose(self, img1, img2):
        """recover_pose: Recovers the rotation and translation vectors for camera 1 to camera 2 in camera 1's reference frame
        
        NOTE: The translation will only be known up to a scale 
              Additional information needed to recover the scale
        
        https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#gadb7d2dfcc184c1d2f496d8639f4371c0 --> recover pose function

        Args:
            img1 (numpy array): _description_
            img2 (numpy array): _description_

        Returns:
            numpy array: rotation matrix from image 1 to image 2 in image 1 reference frame
            numpy array: translation vector 
        """
        E, p1, p2 = self.calc_essential(img1, img2)
        _, R, t, _ = cv2.recoverPose(E, p1, p2, cameraMatrix=self.K)
        return R, t
    
    
    #methods used in Robot Autonomy
    def recover_pose(self, points, global_positions):
        """recover_pose: a number of points are needed: Both 3D corresponding points and 2D image points are needed and must be in the correct order
        
        AS IMPLEMENTED, THIS FUNCTION ASSUMES POINTS1, POINTS2, AND GLOBAL_POSITIONS ARE IN THE SAME ORDER
        POINTS1: nx2 matrix
        POINTS2: nx2 matrix
        GLOBAL_POSITIONS: nx2 matrix --> we assume that all points on the surface of the moon are at Z = 0
        
        NOTE: The translation will only be known up to a scale 
              Additional information needed to recover the scale
        """
        n = global_positions.shape[0]
        
        #trying to recreate the m matrix from https://drive.google.com/drive/u/0/folders/1ycJFE38fdjRI5dL0B4rkTjIrNyveUCx5
        uX = points[:,0:1]*global_positions[:,0:1]
        uY = points[:,0:1]*global_positions[:,1:2]
        vX = points[:,1:2]*global_positions[:,0:1]
        vY = points[:,1:2]*global_positions[:,1:2]
        u = points[:,0:1]
        v = points[:,1:2]
        encoded_pos = np.hstack((global_positions, np.ones((n,1)), np.zeros((n,3)), uX, uY, u, v)) #n x 9 matrix 
        
        encoded_pos = np.repeat(encoded_pos, 2, axis=0) #2nx9 matrix 
        
        row_1 = np.hstack([-1*np.ones((1,3)), np.zeros((1,3)), np.ones((1,3))])
        row_2 = np.hstack([np.zeros((1,3)), -1*np.ones((1,3)),  np.ones((1,3))])
        M = np.vstack((row_1, row_2))
        M = np.tile(M, (n,1)) #2nx9 matrix 
        
        M = M*encoded_pos #element wise multiplication 
        U, S, Vh = np.linalg.svd(M)
        
        V = Vh.T
        H = V[:,-1].reshape(3,3) #extract last column and reshape
        Kinv = np.linalg.inv(self.K)
        scale = np.linalg.norm((Kinv@H)[:,0]) #take norm of the first column of K^-1 @ H
        r0 = 1 / scale * (Kinv@H)[:,0].reshape(-1,1)
        r1 = 1 / scale * (Kinv@H)[:,1].reshape(-1,1)
        r2 = np.cross(r0, r1).reshape(-1, 1)
        R = np.vstack((r0.T, r1.T, r2.T))
        t = 1/scale (Kinv@H)[:,2].reshape(-1,1)
        return R, t 
    
    def get_scale(self, tile, lat, deg=True):
        if deg:
            lat = np.deg2rad(lat)
            
        predictions = self.detector.detect_craters(tile)
        diameter = (predictions[0,2] / np.cos(lat) + predictions[0,3]) / 2 #need to scale radius
        radius_image = diameter/2 
        # radius_image = self.detector.estimate_crater_radius(tile)[0] #radius in pixels only get the first one
        radius_real = radius_image / tile.image.shape[0] * tile.win * 100      #voodoo magic to get real radius
        z = self.focal * radius_real / radius_image
    
        return z
    
    def get_position_global(self, tile):    
        predictions = self.detector.detect_craters(tile)
        
        image_points = predictions[0,:2].flatten()
        crater_x, crater_y = image_points
        
        image_points = np.append(image_points, 1).reshape(-1,1) #put in homogeneous form
        
        gu, gv = moon.locate_crater(tile, crater_x,crater_y)
        global_points = np.array([gu,gv])
        
        lat, lon = pixel_to_lat_lon(global_points, deg=True)
        global_points3D = lla2mcmf(lon, lat, 0)*1e3 #converting points to m space
        global_points3D = global_points3D.reshape(-1,1)
        
        
        scale = self.get_scale(tile, lat, deg=True)
        print('Scale', scale)
        
        print(f"Latitude of Crater: {lat}")
        print(f"Longitude of Crater: {lon}")
        t = global_points3D - scale * np.linalg.inv(self.K)@image_points
        t = mcmf2lla(t.flatten()*1e-3)
        
        return t #return tau as tx, ty, tz
        
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
    print(cam.get_position_global(tile))

        
        
        