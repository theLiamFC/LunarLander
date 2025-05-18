from LunarRender import LunarRender
import cv2
import matplotlib.pyplot as plt

moon = LunarRender('WAC_ROI', fov=45)
tile1 = moon.render_m(x=-5000, y=30000, alt=500000)
tile2 = moon.render_m(x=-10000, y=50000, alt=500000)
moon.tile2jpg(tile1, 'lunar_images/tile1.jpg')
moon.tile2jpg(tile2, 'lunar_images/tile2.jpg')

tile1 = cv2.imread('lunar_images/tile1.jpg', cv2.IMREAD_GRAYSCALE)
tile2 = cv2.imread('lunar_images/tile2.jpg', cv2.IMREAD_GRAYSCALE)

orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(tile1, None)
kp2, des2 = orb.detectAndCompute(tile2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1, des2)

matches = sorted(matches, key=lambda x: x.distance)

N = 20

img_matches = cv2.drawMatches(tile1, kp1, tile2, kp2, matches[:N], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Show the matches using matplotlib
plt.figure(figsize=(12, 6))
plt.imshow(img_matches, cmap='gray')
plt.title(f'Top {N} ORB Matches')
plt.axis('off')
plt.show()