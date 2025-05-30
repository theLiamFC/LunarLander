"""
LunarRender Module

Provides the LunarRender class for assembling and rendering composite views
of the Moon’s surface from LROC WAC imagery.

http://pds.lroc.asu.edu/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/EXTRAS/BROWSE/WAC_ROI/WAC_ROI_NEARSIDE_DAWN/

Loads all Geo-referenced tiles in a folder, computes a square camera footprint
based on altitude and field-of-view, stitches overlapping fragments into a single
numv array, and offers utilities to export the result as a JPEG.

Assumes equirectangular projection of input images and does not perform any
advanced manipulation to correct for camera properties. Rendered images will
likely get more distorted further from the equator.

https://en.wikipedia.org/wiki/Equirectangular_projection
"""

import os
from typing import NamedTuple
import numpy as np
import rasterio
from affine import Affine
from rasterio.windows import Window, intersection, from_bounds, transform
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from PIL import Image

MOON_RADIUS_M = 1_737_400 # radius of moon in meters

class Tile(NamedTuple):
    image: np.ndarray
    u: int # global x at center of image in pixels
    v: int # global y at center of image in pixels
    win: int # window size in m
    time: float # simulation time of render

class LunarRender:
    def __init__(self, folder_path, foc=21e-3, size=512, debug=False):
        """
        Initiates the LunarRender class.

        Parameters
        ----------
        folder_path : string
            A string indicating the path to the folder containing the images.
        fov : int
            The simulated field of view (FOV) of the camera in degrees.
        w : int
            The width of the output image in pixels.
        h : int
            The height of the output image in pixels.
        """
        self.folder_path = folder_path
        self.size = size # square pixel size of output image
        self.foc = foc # focal length: m
        self.p_size = 24e-6 # pixel size in m
        self.s_dim = self.p_size * 1024 # sensor dimension: m
        self.fov = 2 * np.arctan(self.s_dim / (2 * self.foc)) # radians
        self.images = {} # dict of images in folder_path
        self.min_max = [0,0,0,0]
        self.debug = debug
        self.verbose = True

        i=0
        for fname in os.listdir(self.folder_path):
            if not fname.lower().endswith('.xml'):
                continue

            base = os.path.splitext(fname)[0]
            img_name = base + '.IMG'
            img_path = os.path.join(self.folder_path, img_name)

            src = rasterio.open(img_path)

            if self.debug:
                print(f"CRS: {src.crs}")
                print(f"Transform: {src.transform}")
                print(f"Bounds: {src.bounds}")
                print(f"Width: {src.width}, Height: {src.height}")
                
                if src.crs:
                    units = src.crs.linear_units
                    print(f"Linear units: {units}")
                
                print(f"CRS string: {src.crs}")
                
                image_data = src.read()
                print(f"Image shape: {image_data.shape}")

            wrap = src.bounds.left > 2729100.0
            
            self.images[img_path] = {
                'src': src, # rasterio object
                'res': src.res, # m/pix
                'min_lat': None,
                'max_lat': None,
                'min_lon': None,
                'max_lon': None,
                'left': src.bounds.left if not wrap else src.bounds.left -10916400.0,
                'right': src.bounds.right if not wrap else src.bounds.right -10916400.0,
                'bottom': src.bounds.bottom,
                'top': src.bounds.top,
                'transform': src.transform if not wrap else Affine.translation(-10916400.0, 0) * src.transform
            }
            
            

            self.min_max = [min(self.images[img_path]['left'], self.min_max[0]),
                            min(self.images[img_path]['bottom'], self.min_max[1]),
                            max(self.images[img_path]['right'], self.min_max[2]),      
                            max(self.images[img_path]['top'], self.min_max[3])]

    def __del__(self):
        """
        Closes all rasterio objects when the class is destroyed.
        """
        for img_file in self.images.values():
            try:
                img_file['src'].close()
            except Exception:
                pass

    def render_ll(self, lon, lat, alt, time=0.0, deg=False):
        """
        Render a composite view centered at (lon, lat) from a given altitude (m).

        This method projects a square camera footprint of size determined by `alt` and the
        field‐of‐view (`self.fov`) onto the ground, finds all images in `self.images` that
        overlap that footprint, reads and resamples those overlapped regions, and stitches
        them into a single `self.size` × `self.size` numv array.

        Parameters
        ----------
        lon : float
            The longitude coordinate of the view center in degrees.
        lat : float
            The latitude coordinate of the view center in degrees.
        alt : float
            Altitude (distance above the map plane) in meters. Controls
            the ground footprint size via `alt * tan(fov/2)`.

        Returns
        -------
        numv.ndarray
            A 2D float32 array of shape `(self.size, self.size)` containing the stitched image.

        Raises
        ------
        ValueError
            If any part of the requested view lies outside the spatial extent of all provided images.
        """
        if deg:
            lon = np.radians(lon)
            lat = np.radians(lat)

        u = MOON_RADIUS_M * lon / 100
        v = MOON_RADIUS_M * lat / 100

        return self.render(u, v, alt, time)

    def render(self, u, v, alt, time=0.0):
        """
        Render a composite view centered at (x, y) from a given altitude.

        This method projects a square camera footprint of size determined by `alt` and the
        field‐of‐view (`self.fov`) onto the ground, finds all images in `self.images` that
        overlap that footprint, reads and resamples those overlapped regions, and stitches
        them into a single `self.size` × `self.size` numv array.

        Parameters
        ----------
        u : float
            The x‐coordinate of the view center in pixels.
        v : float
            The y‐coordinate of the view center in pixels.
        alt : float
            Altitude (distance above the map plane) in meters. Controls
            the ground footprint size via `alt * tan(fov/2)`.

        Returns
        -------
        numv.ndarray
            A 2D float32 array of shape `(self.size, self.size)` containing the stitched image.

        Raises
        ------
        ValueError
            If any part of the requested view lies outside the spatial extent of all provided images.
        """
        render = np.full((self.size, self.size), np.nan, dtype=np.float32) # blank image

        half = alt * np.tan(self.fov / 2.0) # calculate fov coverage in meter

        minx, maxx = u*100 - half, u*100 + half # global coverage in x pixels
        miny, maxy = v*100 - half, v*100 + half # global coverage in y pixels

        count = 0
        for meta in self.images.values(): # loop through all base images
            src = meta['src']

            # Check for overlap between fov coverage and current base image
            ovrlp_l = max(minx, meta['left'])
            ovrlp_r = min(maxx, meta['right'])
            ovrlp_b = max(miny, meta['bottom'])
            ovrlp_t = min(maxy, meta['top'])

            if ovrlp_l >= ovrlp_r or ovrlp_b >= ovrlp_t:
                continue # skip base image if no overlap
            count += 1

            # get pixel window from overlap coverage in meters
            win = from_bounds(
                ovrlp_l, ovrlp_b, ovrlp_r, ovrlp_t,
                transform=meta['transform']
            )
            win = intersection(win, Window(0, 0, src.width, src.height))
            
            # get final pixel dimensions of this fragment
            frac_w = (win.width  * src.res[0]) / (2 * half)
            frac_h = (win.height * src.res[1]) / (2 * half)
            out_w  = int(np.ceil(frac_w * self.size)) # round to extra pixel
            out_h  = int(np.ceil(frac_h * self.size)) # round to extra pixel

            # get fragment from base image
            fragment = src.read(
                1,
                window=win,
                out_shape=(out_h, out_w),
                resampling=Resampling.bilinear
            )
            
            # locate fragment in the tile
            tlx, tly = transform(win, meta['transform']) * (0, 0)
            col_off = int(((tlx - minx) / (2 * half)) * self.size)
            row_off = int(((maxy - tly) / (2 * half)) * self.size)

            # clip fragment size to fit in tile
            if col_off + out_w > self.size:
                out_w = self.size - col_off
            if row_off + out_h > self.size:
                out_h = self.size - row_off

            render[ # insert fragment into tile
                row_off : row_off + out_h,
                col_off : col_off + out_w
            ] = fragment[:out_h, :out_w]

        if np.isnan(render).any():
            raise ValueError(f"Requested render at {u,v,alt} out of bounds of available imaging: min {self.min_max[0:2]}, max {self.min_max[2:4]}")
        else:
            if self.verbose: print(f"Rendered {render.shape[0]}x{render.shape[1]} image at {u,v,alt} (px,px,m) from {count} images in {self.folder_path}")
            
            # normalize values to 0-255
            minv, maxv = render.min(), render.max()
            if maxv > minv:
                norm = (render - minv) / (maxv - minv)
            else:
                norm = np.zeros_like(render)
            tile_uint8 = (norm * 255).astype(np.uint8)
        
        return Tile(image=tile_uint8, u=u, v=v, win=2*half/100, time=time)
    
    def tile2jpg(self, tile, filename):
        """
        Converts a 2D numv array of image data into a JPEG file,
        creating any missing directories in the output path.

        Parameters
        ----------
        tile : 2D numv array
            A 2D numv array containing image data.
        filename : string
            The desired path and name of the output JPEG file.
        """
    
        # Ensure output directory exists
        out_dir = os.path.dirname(filename)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        # Save as JPEG
        img = Image.fromarray(tile.image, mode='L')
        img.save(filename, format='JPEG', quality=90)

def locate_crater(tile, u, v):
    """
    Translates pixel coordinates within a given tile to global coordinates
    in pixels. Assumes image axes aligned with the origin in the top left.

    Parameters
    ----------
    tile : Tile class
        A custom class containing an 2d np array, centering x,y (m), and window size (m).
    u : float
        The pixel coordinates in x axis.
    v : float
        The pixel coordinates in y axis.
    
    Returns
    ----------
    u, v : float
        The global coordinates in pixels.
    """
    # Calculate fractional offset from center
    x_offset_f = (u / (tile.image.shape[0]-1)) - 0.5
    y_offset_f = 0.5 - (v / (tile.image.shape[1]-1))

    # add frac * win to x,y to calc global position within tile
    gu = tile.u + x_offset_f * tile.win
    gv = tile.v + y_offset_f * tile.win

    return gu,gv

def pixel_to_lat_lon(pixel, pixel_scale=100, deg=False):
    u,v = pixel.flatten()
    x = pixel_scale*u
    y = pixel_scale*v
    lon = x / MOON_RADIUS_M
    lat = y / MOON_RADIUS_M
    
    if deg:
        lat = np.degrees(lat)
        lon = np.degrees(lon)
        
    return lat, lon

def lat_lon_to_pixel(lat, lon, pixel_scale=100, deg=False):
    if deg:
        lat = np.radians(lat)
        lon = np.radians(lon)
        
    y = lat * MOON_RADIUS_M
    x = -lon * MOON_RADIUS_M
    
    u = x / pixel_scale
    v = y / pixel_scale
    
    return u, v
    
# Example usage:
if __name__ == "__main__":
    moon = LunarRender('../WAC_ROI',debug=True)
    tile = moon.render(u=-900, v=17000, alt=100000)
    moon.tile2jpg(tile, "lunar_images/tile.jpg")


