from camera import Camera

cam = Camera()
K = cam.get_K()
print(cam.get_K()) #returns the camera intrinsic matrix

from lunar_render import LunarRender
from crater_detector import CraterDetector

moon = LunarRender('../WAC_ROI',debug=False)
# tile = moon.render(u=0, v=0, alt=50000)
tile = moon.render_ll(lat=0.5,lon=20,alt=100000,deg=True)

detector = CraterDetector()
# detector.view_craters(tile)
print(cam.get_position_global(tile, 100000)) # ouputs lat, lon, altitide (deg, deg, km) of camera position in world frame