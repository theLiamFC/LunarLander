import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from PIL import Image

class LunarMosaicDeg:
    def __init__(self, img_path, pix_per_deg, ref_altitude):
        """
        img_path      : path to .IMG
        pix_per_deg   : map sampling in pixels per degree
        ref_altitude  : altitude (m) at which 1 pixel => r0 m on ground
        """
        self.ds = rasterio.open(img_path)
        # lunar radius
        R = 1_737_400  
        # meters per degree at equator
        m_per_deg = 2 * np.pi * R / 360  
        # base resolution [m/pixel]
        self.r0 = m_per_deg / pix_per_deg  
        self.z0 = ref_altitude
        self.transform = self.ds.transform

    def get_tile(self, x, y, w, h, z):
        """
        x, y : map coords (deg longitude, deg latitude)
        w, h : output size in pixels
        z    : current altitude in meters
        """
        # 1) Convert lon/lat deg to pixel indices
        col_f, row_f = ~self.transform * (x, y)
        col, row = int(col_f), int(row_f)

        # 2) Compute zoom scale
        s = max(z / self.z0, 0.1)

        # 3) Window in native pixels
        win_w = int(round(w * s))
        win_h = int(round(h * s))
        window = Window(col - win_w//2, row - win_h//2, win_w, win_h)

        # 4) Read & downsample to (h,w)
        tile = self.ds.read(
            1,
            window=window,
            out_shape=(h, w),
            resampling=Resampling.bilinear
        )
        return tile
    
    def tile2jpg(self, tile, filename):
        minv, maxv = tile.min(), tile.max()
        if maxv > minv:
            norm = (tile - minv) / (maxv - minv)
        else:
            norm = np.zeros_like(tile)

        # --- 3) Convert to uint8 ---
        tile_uint8 = (norm * 255).astype(np.uint8)

        # --- 4) Create a PIL Image in mode "L" (8-bit grayscale) ---
        img = Image.fromarray(tile_uint8, mode='L')

        # --- 5) Save as JPEG ---
        img.save(filename, format='JPEG', quality=90)

# Usage:
# If your WAC_GLOBAL mosaic is 128 pixels/degree [6],
# and you choose z0 = 1000 m:
mosaic = LunarMosaicDeg('WAC_ROI_NORTH_SUMMER/WAC_ROI_NORTH_SUMMER_256P.IMG', pix_per_deg=128, ref_altitude=1000)
tile_near = mosaic.get_tile(x=10.0, y=20.0, w=512, h=512, z=500)
tile_far  = mosaic.get_tile(x=1000.0, y=1000.0, w=512, h=512, z=10000)

mosaic.tile2jpg(tile_near, 'near.jpg')
mosaic.tile2jpg(tile_far, 'far.jpg')
