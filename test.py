# Example usage:

from lunar_render import LunarRender
from crater_detector import CraterDetector

moon = LunarRender('WAC_ROI', fov=45)
tile = moon.render_m(x=-2000, y=-50000, alt=50000)
# moon.tile2jpg(tile, "lunar_images/tile.jpg")
detector = CraterDetector()

detector.view_craters(tile)

