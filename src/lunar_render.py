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
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

MOON_RADIUS_M = 1_737_400 # radius of moon in meters

class Tile(NamedTuple):
    image: np.ndarray
    u: int # global x at center of image in pixels
    v: int # global y at center of image in pixels
    win: int # window size in m
    time: float # simulation time of render

class LunarRender:
    def __init__(self, folder_path, foc=21e-3, size=1080, debug=False):
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

        for fname in os.listdir(self.folder_path):
            print(f"Processing {fname}...") if self.debug else None
            if not fname.lower().endswith('.img'):
                print(f"Skipping {fname}, not a valid image file.") if self.debug else None
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

            wrap = src.bounds.left > 2729100.0 if fname.startswith('WAC_ROI_NEAR') else False
            
            self.images[img_path] = {
                'src': src, # rasterio object
                'res': src.res, # m/pix
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
            
            # for plotting trajectory
            self.MIN_RATIO = 1.0 
            self.MAX_DIM = 6000 

    def __del__(self):
        """
        Closes all rasterio objects when the class is destroyed.
        """
        for img_file in self.images.values():
            try:
                img_file['src'].close()
            except Exception:
                pass

    def render_ll(self, lat, lon, alt, time=0.0, deg=False):
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

        u = MOON_RADIUS_M * lon / 100 + (2_729_100.0 / 100)
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
            raise ValueError(f"Requested render at {u,v,alt} (px,px,m) out of bounds of available imaging: min {self.min_max[0:2]}, max {self.min_max[2:4]}")
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
    
    def render_trajectory(self, trajectory_lla):
        MOON_RADIUS_M = 1_737_400  # Moon radius in meters
        
        # Initialize trajectory array
        trajectory_xyz = np.zeros((len(trajectory_lla), 3))
        
        # Convert LLA to XYZ with proper coordinate scaling
        for i, (lat, lon, alt) in enumerate(trajectory_lla):
            x = MOON_RADIUS_M * np.radians(lon) + 2_729_100.0
            y = MOON_RADIUS_M * np.radians(lat)
            trajectory_xyz[i] = [x, y, alt]
        
        # Calculate bounds with 10% padding
        mins = np.min(trajectory_xyz, axis=0)
        maxs = np.max(trajectory_xyz, axis=0)
        
        x_pad = (maxs[0] - mins[0]) * 0.1
        y_pad = (maxs[1] - mins[1]) * 0.1
        minx, maxx = mins[0] - x_pad, maxs[0] + x_pad
        miny, maxy = mins[1] - y_pad, maxs[1] + y_pad

        # Calculate natural dimensions
        natural_width = maxx - minx
        natural_height = maxy - miny

        # Adjust coordinate bounds to enforce aspect ratio
        if natural_height / natural_width > self.MIN_RATIO:
            # Too tall - expand width
            target_width = natural_height / self.MIN_RATIO
            width_expand = (target_width - natural_width) / 2
            minx -= width_expand
            maxx += width_expand
        elif natural_width / natural_height > self.MIN_RATIO:
            # Too wide - expand height  
            target_height = natural_width / self.MIN_RATIO
            height_expand = (target_height - natural_height) / 2
            miny -= height_expand
            maxy += height_expand

        # Recalculate dimensions after aspect ratio adjustment
        final_width = maxx - minx
        final_height = maxy - miny

        # Scale down if exceeding maximum dimension
        scale_factor = min(self.MAX_DIM/final_width, self.MAX_DIM/final_height)
        if scale_factor < 1:
            final_width *= scale_factor
            final_height *= scale_factor

        canvas_width = int(np.ceil(final_width))
        canvas_height = int(np.ceil(final_height))
        
        # Create output canvas
        render = np.full((canvas_height, canvas_width), np.nan, dtype=np.float32)
        
        # Image composition logic
        count = 0
        for meta in self.images.values():
            src = meta['src']
            
            # Calculate overlap
            ovrlp_l = max(minx, meta['left'])
            ovrlp_r = min(maxx, meta['right'])
            ovrlp_b = max(miny, meta['bottom'])
            ovrlp_t = min(maxy, meta['top'])
            
            if ovrlp_l >= ovrlp_r or ovrlp_b >= ovrlp_t:
                continue
                
            # Calculate pixel window
            win = from_bounds(ovrlp_l, ovrlp_b, ovrlp_r, ovrlp_t,
                            transform=meta['transform'])
            win = intersection(win, Window(0, 0, src.width, src.height))
            
            # Calculate fragment dimensions using ADJUSTED bounds
            frag_width = int(np.ceil((ovrlp_r - ovrlp_l) / (maxx - minx) * canvas_width))
            frag_height = int(np.ceil((ovrlp_t - ovrlp_b) / (maxy - miny) * canvas_height))
            
            # Skip tiny fragments
            if frag_width < 1 or frag_height < 1:
                continue
                
            # Read and resample fragment
            fragment = src.read(1, window=win, 
                            out_shape=(frag_height, frag_width),
                            resampling=Resampling.bilinear)
            
            # Calculate position using ADJUSTED bounds
            col_start = int((ovrlp_l - minx) / (maxx - minx) * canvas_width)
            row_start = int((maxy - ovrlp_t) / (maxy - miny) * canvas_height)
            
            # Strict bounds checking
            col_end = min(col_start + frag_width, canvas_width)
            row_end = min(row_start + frag_height, canvas_height)
            col_start = max(col_start, 0)
            row_start = max(row_start, 0)
            
            if col_end > col_start and row_end > row_start:
                actual_height = row_end - row_start
                actual_width = col_end - col_start
                render[row_start:row_end, col_start:col_end] = fragment[:actual_height, :actual_width]
                count += 1

        # Better handling of missing data
        if count == 0:
            raise ValueError("No valid image data found for trajectory bounds")
            
        # Replace NaNs with zeros
        render = np.nan_to_num(render, nan=0.0)
            
        # Normalize to 0-255
        if render.max() > render.min():
            render = (render - render.min()) / (render.max() - render.min()) * 255
        
        return Tile(image=render.astype(np.uint8), u=-1, v=-1, win=-1, time=-1)

    def plot_trajectory_with_image(self, trajectory_lla, figsize=(12, 8), dpi=100):
        """
        Creates a matplotlib plot with trajectory overlaid on rendered lunar surface image.
        
        Args:
            trajectory_lla: List of (lat, lon, alt) points
            figsize: Figure size tuple (width, height)
            dpi: Figure resolution
        
        Returns:
            fig, ax: Matplotlib figure and axes objects
        """
        MOON_RADIUS_M = 1_737_400
        
        # Get the rendered image using existing function
        tile = self.render_trajectory(trajectory_lla)
        image = tile.image
        
        # Convert trajectory to XYZ coordinates (same as in render function)
        trajectory_xyz = np.zeros((len(trajectory_lla), 3))
        for i, (lat, lon, alt) in enumerate(trajectory_lla):
            x = MOON_RADIUS_M * np.radians(lon) + 2_729_100.0
            y = MOON_RADIUS_M * np.radians(lat)
            trajectory_xyz[i] = [x, y, alt]
        
        # Calculate the same bounds used in rendering
        mins = np.min(trajectory_xyz, axis=0)
        maxs = np.max(trajectory_xyz, axis=0)
        
        x_pad = (maxs[0] - mins[0]) * 0.1
        y_pad = (maxs[1] - mins[1]) * 0.1
        minx, maxx = mins[0] - x_pad, maxs[0] + x_pad
        miny, maxy = mins[1] - y_pad, maxs[1] + y_pad
        
        # Apply same aspect ratio adjustments as in render function
        natural_width = maxx - minx
        natural_height = maxy - miny
        
        if natural_height / natural_width > self.MIN_RATIO:
            target_width = natural_height / self.MIN_RATIO
            width_expand = (target_width - natural_width) / 2
            minx -= width_expand
            maxx += width_expand
        elif natural_width / natural_height > self.MIN_RATIO:
            target_height = natural_width / self.MIN_RATIO
            height_expand = (target_height - natural_height) / 2
            miny -= height_expand
            maxy += height_expand
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Display the image with proper coordinate extent
        extent = [minx, maxx, miny, maxy]
        ax.imshow(image, cmap='gray', extent=extent, origin='upper', aspect='equal')

        # Process craters
        crater_df = pd.read_csv("crater_logs_noisy_05.csv")
        unique_craters = set()
        for _, row in crater_df.iterrows():
            tile = self.render_ll(lat=row['True_X'],lon=row['True_Y'],alt=row['True_Z'], deg=True)
            for i in range(1, 11):
                conf_col = f'Crater{i}_Conf'
                u_col = f'Crater{i}_U'
                v_col = f'Crater{i}_V'
                
                # Check if columns exist and have valid data
                if conf_col not in row or pd.isna(row[conf_col]):
                    continue
                    
                if row[conf_col] > 0.5:  #Confidence threshold
                    try:
                        u = row[u_col]
                        v = row[v_col]
                        gu, gv = locate_crater(tile, u, v)

                        gx = 100 * gu
                        gy = 100 * gv

                        cmap = plt.cm.plasma  # Yellow (low) to Red (high)
                        norm_confidence = (row[conf_col] - 0.5) / 0.5  # Normalize 0.4-1.0 → 0-1
                        color = cmap(norm_confidence)
                        
                        ax.scatter(gx, gy, marker='x', color=color, 
                                s=50, alpha=0.8, zorder=3,
                                edgecolors='black', linewidth=0.5)

                        # ax.scatter(gx, gy, marker='x', color='cyan', 
                        #             s=50, alpha=0.7, zorder=3)
                        
                        print(f"Found crater at {gu},{gv}")
                        
                        # # De-duplicate (5m grid)
                        # key = (round(gu/5)*5, round(gv/5)*5)
                        # if key not in unique_craters:
                        #     unique_craters.add(key)
                        #     ax.scatter(gu, gv, marker='x', color='cyan', 
                        #             s=50, alpha=0.7, zorder=3)
                    except KeyError:
                        continue
                    except Exception as e:
                        print(f"Error processing crater: {e}")
                        continue
        
        # Plot trajectory points
        x_coords = trajectory_xyz[:, 0]
        y_coords = trajectory_xyz[:, 1]
        
        # Plot trajectory line
        ax.plot(x_coords, y_coords, 'r-', linewidth=2, label='Trajectory', alpha=0.8)
        
        # Plot start and end points
        ax.scatter(x_coords[0], y_coords[0], c='green', s=100, marker='o', 
                label='Start', zorder=5, edgecolors='white', linewidth=2)
        ax.scatter(x_coords[-1], y_coords[-1], c='red', s=100, marker='s', 
                label='End', zorder=5, edgecolors='white', linewidth=2)
        
        # Plot intermediate waypoints
        # if len(x_coords) > 2:
        #     ax.scatter(x_coords[1:-1], y_coords[1:-1], c='yellow', s=50, marker='o', 
        #             label='Waypoints', zorder=4, edgecolors='black', linewidth=1)
        
        # Set labels and formatting
        ax.set_xlabel('X Coordinate (meters)', fontsize=12)
        ax.set_ylabel('Y Coordinate (meters)', fontsize=12)
        ax.set_title('Crater Detection Trajectory Overlay', fontsize=14, fontweight='bold')
        
        # Format axes with scientific notation for large numbers
        ax.ticklabel_format(style='scientific', axis='both', scilimits=(0,0))
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add legend
        ax.legend(loc='upper right', framealpha=0.8)
        
        # Ensure equal aspect ratio
        ax.set_aspect('equal', adjustable='box')

        # After all scatter plotting:
        norm = Normalize(vmin=0.5, vmax=1.0)
        sm = ScalarMappable(norm=norm, cmap=plt.cm.plasma)
        sm.set_array([])  # Only needed for older matplotlib versions

        # Add colorbar to the current axes
        cbar = plt.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label('Crater Confidence', fontsize=12)
        cbar.set_ticks([0.5, 0.75, 1.0])
        cbar.ax.tick_params(labelsize=10)
        
        # Tight layout
        plt.tight_layout()
        
        return fig, ax

    def save_trajectory_plot(self, trajectory_lla, filename, figsize=(12, 8), dpi=300):
        """
        Convenience function to save trajectory plot directly to file.
        
        Args:
            trajectory_lla: List of (lat, lon, alt) points
            filename: Output filename (e.g., 'trajectory_plot.png')
            figsize: Figure size tuple
            dpi: Output resolution
        """
        fig, ax = self.plot_trajectory_with_image(trajectory_lla, figsize, dpi)
        fig.savefig(filename, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"Trajectory plot saved to {filename}")

    
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
        print(f"Saved image as {filename}")

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
    x_offset_f = (u / tile.image.shape[0]) - 0.5
    y_offset_f = 0.5 - (v / tile.image.shape[1])

    # add frac * win to x,y to calc global position within tile
    gu = tile.u + x_offset_f * tile.win
    gv = tile.v + y_offset_f * tile.win

    return gu, gv

def pixel_to_lat_lon(u, v, pixel_scale=100, deg=False):
    
    x = pixel_scale*(u - (2_729_100.0 / 100))
    y = pixel_scale*v
    lon = x / MOON_RADIUS_M
    lat = y / MOON_RADIUS_M
    
    if deg:
        lon = np.degrees(lon)
        lat = np.degrees(lat)
        
    return lat, lon

def lat_lon_to_pixel(lat, lon, pixel_scale=100, deg=False):
    if deg:
        lat = np.radians(lat)
        lon = np.radians(lon)
        
    u = MOON_RADIUS_M * lon / 100 + (2_729_100.0 / 100)
    v = MOON_RADIUS_M * lat / 100
    
    return u, v
    
# Example usage:
if __name__ == "__main__":
    moon = LunarRender('WAC_ROI',debug=False)
    # tile = moon.render(u=28153, v=-18194, alt=100000)
    # tile = moon.render_ll(lon=0.0127, lat=17.34, alt=3212, deg=True)
    # moon.tile2jpg(tile, "lunar_images/tile.jpg")

    trajectory_lla = np.genfromtxt("traj_all.csv",delimiter=",",skip_header=True)[:,0:3]

    # traj_tile = moon.render_trajectory(trajectory_lla)
    # moon.tile2jpg(traj_tile, "lunar_images/traj_tile.jpg")

    fig, ax = moon.plot_trajectory_with_image(trajectory_lla)
    plt.show()

# WAC_ROI_FARSIDE_DUSK:
# IMAGE_NAME	    LEFT (m)	RIGHT (m)	BOTTOM (m)	TOP (m)
# E300S1350_100M	2,729,100	-1,819,400	5,458,200	0
# E300N2250_100M	5,458,200	0	        8,187,300	1,819,400
# E300S2250_100M	5,458,200	-1,819,400	8,187,300	0
# E300N1350_100M	2,729,100	0	        5,458,200	1,819,400

# WAC_ROI_NEARSIDE_DAWN:
# IMAGE_NAME        LEFT (m)    BOTTOM (m)  RIGHT (m)   TOP (m)
# E300S0450_100M    0           -1,819,400  2,729,100   0
# E300N0450_100M    0           0           2,729,100   1,819,400
# E300S3150_100M    8,187,300   -1,819,400  10,916,400  0
# E300N3150_100M    8,187,300   0           10,916,400  1,819,400
