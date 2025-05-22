import cv2
import numpy as np

"""

NOTES:
1. Camera object should initialize a crater_detector object inside it --> will be needed to calculate translational information 
2. Methods still need to be tested. Basic stuff is in there and the steps are written out with steps and sources.
3. Camera matrix needs to be created/talked about



"""


class Camera():
    def __init__(self, K):
        self.orb = cv2.ORB_create()
        self.K = K #K represents the camera intrinsic matrix --> this needs to be input

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
    
