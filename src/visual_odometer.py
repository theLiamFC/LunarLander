import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import NamedTuple

from lunar_render import Tile

class VisualOdometer:
    def __init__(self):
        """
        Initiates the Visual Odometer Class.

        Parameters
        ----------
        """
        self.orb = cv2.ORB_create()
    
    def get_velocity(self, state1, state2, tile1, tile2, plot=False):
        """
        Calculate velocity using ORB feature matching between two images.
        
        Parameters
        ----------
        state1, state2 : array-like
            State vectors containing [x, y, z, ...] where z is height
        tile1, tile2 : Tile
            Tile objects containing images and metadata
            
        Returns
        -------
        tuple
            (vx, vy) velocities in x and y directions
        """
        # Extract images
        img1 = tile1.image
        img2 = tile2.image
        
        # Detect ORB keypoints and descriptors
        kp1, des1 = self.orb.detectAndCompute(img1, None)
        kp2, des2 = self.orb.detectAndCompute(img2, None)
        
        assert des1 is not None and des2 is not None, "Descriptors are None, check image inputs"
        
        # Match features using BFMatcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        
        assert len(matches) > 4, "Not enough matches found to compute homography"
        
        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)

        if plot:
            print("Plotting matches...")
            img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            plt.figure(figsize=(15, 8))
            plt.imshow(img_matches, cmap='gray')
            plt.title(f'Feature Matches ({len(matches)} total)')
            plt.axis('off')
            plt.show()
        
        # Extract matched keypoints
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:20]]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:20]]).reshape(-1, 1, 2)
        
        # Find homography matrix
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        assert H is not None, "Homography matrix could not be computed"
        
        # Extract rotation and translation from homography
        # Decompose homography assuming planar motion
        # For small rotations, we can approximate the translation
        
        # Get the translation components from homography
        # H[0,2] and H[1,2] represent pixel displacement
        dx_pixels = H[0, 2]
        dy_pixels = H[1, 2]

        print(f"Pixel Displacement: dx={dx_pixels}, dy={dy_pixels}")
        
        # Convert pixel displacement to real-world displacement using scale
        # Use height from state vectors as scale factor
        height1 = state1[2]
        height2 = state2[2]
        avg_height = (height1 + height2) / 2.0
        
        # Calculate meters per pixel using window size and image dimensions
        img_size = img1.shape[0]
        meters_per_pixel = tile1.win / img_size
        
        # Scale by height (higher altitude = larger ground coverage per pixel)
        scale_factor = avg_height

        focal_length_pixels = (21 / 24e-3 * 1024) * img_size  # Convert mm to pixels
        gsd = avg_height / focal_length_pixels
        
        # Convert to real-world displacement
        dx_meters = -dx_pixels / 512 * tile1.win
        dy_meters = dy_pixels / 512 * tile1.win
        
        # Calculate time difference
        dt = tile2.time - tile1.time
        
        assert dt > 0, "Time difference must be positive"
        
        # Calculate velocities
        vx = dx_meters / dt
        vy = dy_meters / dt
        
        return vx, vy
