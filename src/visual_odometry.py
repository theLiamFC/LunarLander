import cv2

class VisualOdometry():
    def __init__(self):
        self.orb = cv2.ORB_create()

    def get_odometry(self,tile1,tile2):
        kp1, des1 = self.orb.detectAndCompute(tile1, None)
        kp2, des2 = self.orb.detectAndCompute(tile2, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        N = 20
        img_matches = cv2.drawMatches(tile1, kp1, tile2, kp2, matches[:N], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)