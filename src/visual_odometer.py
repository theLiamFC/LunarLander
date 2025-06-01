import numpy as np
import cv2
import matplotlib.pyplot as plt

from lunar_render import Tile

class VisualOdometer:
    def __init__(self, foc=21e-3):
        """
        Initiates the Visual Odometer Class.

        Parameters
        ----------
        """
        self.foc = foc # focal length: m
        self.p_size = 24e-6 # pixel size in m
        self.s_dim = self.p_size * 1024 # sensor dimension: m
        self.fov = 2 * np.arctan(self.s_dim / (2 * self.foc)) # radians

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
        
        #assert des1 is not None and des2 is not None, "Descriptors are None, check image inputs"
        if des1 is None:
            plt.figure(figsize=(15, 8))
            plt.imshow(img1, cmap='gray')
            plt.title(f'Des1 is None - Image 1')
            plt.axis('off')
            plt.show()
        if des2 is None:
            plt.figure(figsize=(15, 8))
            plt.imshow(img2, cmap='gray')
            plt.title(f'Des2 is None - Image 2')
            plt.axis('off')
            plt.show()
        
        # Match features using BFMatcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        
        if len(matches) < 4: plot = True

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
            
        assert len(matches) >= 4, "Not enough matches found to compute homography"
        
        # Extract matched keypoints
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:20]]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:20]]).reshape(-1, 1, 2)

        tx = np.mean(np.percentile(dst_pts[:,0,0] - src_pts[:,0,0], 1.0))
        ty = np.mean(np.percentile(dst_pts[:,0,1] - src_pts[:,0,1],1.0))
        dx_pixels = tx
        dy_pixels = ty

        print(f"Pixel Displacement: dx={dx_pixels}, dy={dy_pixels}")

        height1 = state1[2]
        height2 = state2[2]
        avg_height = (height1 + height2) / 2.0
        
        meters_per_pixel = 2 * (avg_height/100 * self.p_size) / self.foc
        print(f"meters_per_pixel: {meters_per_pixel}")
                
        # Convert to real-world displacement
        dx_meters = -dx_pixels * meters_per_pixel
        dy_meters = dy_pixels * meters_per_pixel
        
        # Calculate time difference
        dt = tile2.time - tile1.time
        
        assert dt > 0, "Time difference must be positive"
        
        # Calculate velocities
        vx = dx_meters / dt
        vy = dy_meters / dt
        
        return vx, vy

if __name__ == "__main__":
    from lunar_render import LunarRender
    # Example usage
    
    odometer = VisualOdometer()
    moon = LunarRender("WAC_ROI")

    minU = 3000.0
    maxU  = 4000.0
    minV = -500.0
    maxV = 500.0
    alt = 30000.0

    n = 100
    u = np.linspace(minU, maxU, n)
    v = np.linspace(minV, maxV, n)
    altitudes = np.full(n, alt)

    init_tile = moon.render(u=u[0], v=v[0], alt=altitudes[0])
    tile_history = [init_tile]

    vx = np.zeros(n)
    vy = np.zeros(n)
    true_vx = np.zeros(n)
    true_vy = np.zeros(n)

    rando = np.random.randint(0, n-1)
    for i in range(1, n):
        tile1 = tile_history[-1]
        tile2 = moon.render(u=u[i], v=v[i], alt=altitudes[i], time=i)
        tile_history.append(tile2)
        
        # Simulate states (x, y, z) for the two tiles
        state1 = np.array([u[i-1], v[i-1], altitudes[i-1]])
        state2 = np.array([u[i], v[i], altitudes[i]])
        
        # Get velocity
        vx[i], vy[i] = odometer.get_velocity(state1, state2, tile1, tile2, plot=False)
        dt = tile2.time - tile1.time  # Should be 1 if time=i
        true_vx[i] = (u[i] - u[i-1]) / dt
        true_vy[i] = (v[i] - v[i-1]) / dt

        if true_vx[i] - vx[i] > 3000 or true_vy[i] - vy[i] > 3000:
            odometer.get_velocity(state1, state2, tile1, tile2, plot=True)
        elif i == rando:
            print("random check")
            odometer.get_velocity(state1, state2, tile1, tile2, plot=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
    time = np.linspace(0, n-1, n)  # Assuming time is just the index for simplicity

    # X velocity subplot
    ax1.plot(time, true_vx, label='True X Velocity')
    ax1.plot(time, vx, label='Estimated X Velocity', linestyle='--')
    ax1.set_ylabel('Velocity X')
    ax1.legend()
    ax1.grid(True)

    # Y velocity subplot
    ax2.plot(time, true_vy, label='True Y Velocity')
    ax2.plot(time, vy, label='Estimated Y Velocity', linestyle='--')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Velocity Y')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()