# Transformations
import numpy as np

def oe2mci(a, e, i_deg, RAAN_deg, omega_deg, nu_deg):
    mu_moon = 4.9048695e3  # km^3/s^2 (G*M of the Moon)

    # Convert angles from degrees to radians
    i = np.deg2rad(i_deg)
    RAAN = np.deg2rad(RAAN_deg)
    omega = np.deg2rad(omega_deg)
    nu = np.deg2rad(nu_deg)

    # Position in Perifocal (PQW) frame
    r = a * (1 - e**2) / (1 + e * np.cos(nu))
    r_PQW = np.array([
        r * np.cos(nu),
        r * np.sin(nu),
        0
    ])

    # Eccentric anomaly for velocity
    E = 2 * np.arctan2(np.sqrt((1 - e) / (1 + e)) * np.tan(nu / 2), 1)

    # Mean motion
    n = np.sqrt(mu_moon / a**3)

    # Velocity in PQW frame
    v_PQW = (a * n) / (1 - e * np.cos(E)) * np.array([
        -np.sin(E),
        np.sqrt(1 - e**2) * np.cos(E),
        0
    ])

    # Rotation matrix: PQW → MCI
    R = np.array([
        [
            np.cos(RAAN)*np.cos(omega) - np.sin(RAAN)*np.cos(i)*np.sin(omega),
            -np.cos(RAAN)*np.sin(omega) - np.sin(RAAN)*np.cos(i)*np.cos(omega),
            np.sin(RAAN)*np.sin(i)
        ],
        [
            np.sin(RAAN)*np.cos(omega) + np.cos(RAAN)*np.cos(i)*np.sin(omega),
            -np.sin(RAAN)*np.sin(omega) + np.cos(RAAN)*np.cos(i)*np.cos(omega),
            -np.cos(RAAN)*np.sin(i)
        ],
        [
            np.sin(i)*np.sin(omega),
            np.sin(i)*np.cos(omega),
            np.cos(i)
        ]
    ])

    # Convert to MCI frame
    r_MCI = R @ r_PQW
    v_MCI = R @ v_PQW

    return r_MCI, v_MCI

def mci2oe(r_IJK, v_IJK):
    mu_moon = 4.9048695e3  # km^3/s^2 (G*M of the Moon)

    r_norm = np.linalg.norm(r_IJK)
    v_norm = np.linalg.norm(v_IJK)

    h_vec = np.cross(r_IJK, v_IJK)
    h_norm = np.linalg.norm(h_vec)

    W = h_vec / h_norm
    W_i, W_j, W_k = W

    # Inclination
    i = np.arctan2(np.sqrt(W_i**2 + W_j**2), W_k)

    # RAAN
    RAAN = np.arctan2(W_i, -W_j)

    # Semi-major axis
    a = 1 / ((2 / r_norm) - (v_norm**2 / mu_moon))

    # Eccentricity
    p = h_norm**2 / mu_moon
    e = np.sqrt(1 - p / a)

    # Mean motion
    n = np.sqrt(mu_moon / a**3)

    # Eccentric anomaly
    E = np.arctan2(np.dot(r_IJK, v_IJK) / (a**2 * n), (1 - r_norm / a))

    # True anomaly
    nu = 2 * np.arctan2(np.sqrt(1 + e) * np.tan(E / 2), np.sqrt(1 - e))

    # Argument of periapsis
    sin_i = np.sin(i)
    u = np.arctan2(r_IJK[2] / sin_i, r_IJK[0] * np.cos(RAAN) + r_IJK[1] * np.sin(RAAN))
    omega = u - nu

    # Convert angles to degrees
    i_deg = np.degrees(i)
    RAAN_deg = np.degrees(RAAN)
    omega_deg = np.degrees(omega)
    nu_deg = np.degrees(nu)

    # Normalize true anomaly
    if nu_deg < 0:
        nu_deg += 360

    return a, e, i_deg, RAAN_deg, omega_deg, nu_deg

# Convert Moon centered moon fixed coordinates to latitude, longitude, and altitude (meters) (LLA)
def mcmf2lla(MCMF):
    R_moon = 1.7374e6  # meters
    # Assumes spherical Moon (selenocentric coordinates)

    r_x, r_y, r_z = MCMF

    # Compute radial distance from center
    r_norm = np.sqrt(r_x**2 + r_y**2 + r_z**2)

    # Longitude λ (degrees)
    lon_deg = np.degrees(np.arctan2(r_y, r_x))

    # Geocentric latitude ψ (degrees)
    lat_deg = np.degrees(np.arcsin(r_z / r_norm))

    # Altitude above lunar surface (assumes spherical Moon)
    alt_meters= r_norm - R_moon

    return lat_deg, lon_deg, alt_meters

def lla2mcmf(lon_deg, lat_deg, alt_meters):
    R_moon = 1.7374e6  # meters
    
    # Convert angles to radians
    lat_rad = np.deg2rad(lat_deg)
    lon_rad = np.deg2rad(lon_deg)

    # Compute radial distance from Moon's center
    r = R_moon + alt_meters

    # Convert to Cartesian coordinates
    x = r * np.cos(lat_rad) * np.cos(lon_rad)
    y = r * np.cos(lat_rad) * np.sin(lon_rad)
    z = r * np.sin(lat_rad)

    return np.array([x, y, z])

def lla_to_mci(lat_deg, lon_deg, alt_m, t=0.0, theta0=0.0):
    # Convert LLA to Moon-fixed frame (ECEF-like)
    R_moon = 1.7374e6  # meters
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)

    x_fixed = (R_moon + alt_m) * np.cos(lat) * np.cos(lon)
    y_fixed = (R_moon + alt_m) * np.cos(lat) * np.sin(lon)
    z_fixed = (R_moon + alt_m) * np.sin(lat)

    r_fixed = np.array([x_fixed, y_fixed, z_fixed])

    # Rotate to inertial frame (MCI)
    omega_moon = 2 * np.pi / (27.321661 * 86400)
    theta = theta0 + omega_moon * t

    R_z = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])

    r_inertial = R_z @ r_fixed
    return r_inertial

# Get local ENU frame at a point
def landing_frame(lat_deg, lon_deg):
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    up = np.array([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])
    east = np.array([-np.sin(lon), np.cos(lon), 0])
    north = np.cross(up, east)
    return east, north, up

def convert_2d_trajectory_to_3d(trajectory_2d, landing_lat, landing_lon, landing_alt=0.0):
    landing_vec = lla2mcmf(landing_lon, landing_lat, landing_alt)
    east, north, up = landing_frame(landing_lat, landing_lon)

    # Get the final (x, y) offset from the 2D trajectory
    final_offset = trajectory_2d[-1]
    offset_x, offset_y = final_offset

    # Adjust 2D trajectory so that the last point lands at the origin
    # use east so that trajectory is always above equator
    traj_3d = []
    for x, y in trajectory_2d:
        dx = x - offset_x
        dy = y - offset_y
        pos_vec = landing_vec + dx * east + dy * up
        traj_3d.append(pos_vec)

    return np.array(traj_3d), landing_vec

def mci_to_mcmf(r_inertial, t, omega_moon=2 * np.pi / (27.321661 * 86400), theta0=0.0):
    theta = theta0 + omega_moon * t
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R_z = np.array([
        [cos_t,  sin_t, 0],
        [-sin_t, cos_t, 0],
        [0,      0,     1]
    ])
    return R_z @ r_inertial

def convert_traj_to_moon_fixed(traj_inertial):
    traj_fixed = []
    for row in traj_inertial:
        t = row[0]
        r_inertial = row[1:4]
        r_fixed = mci_to_mcmf(r_inertial, t)
        traj_fixed.append(r_fixed)
    return np.array(traj_fixed)

def mcmf_traj_to_lla(traj_fixed_m, R_moon=1737400.0):
    """
    Convert an entire trajectory from Moon-fixed (MCMF) coordinates to LLA (lat, lon, alt).
    
    Parameters:
        traj_fixed_m : np.ndarray
            Shape (N, 3) or (N, >=3) array of (x, y, z) positions in meters in Moon-fixed frame.
        R_moon : float
            Radius of the Moon in meters. Default is 1737400.0 m.
    
    Returns:
        lla_array : np.ndarray
            Shape (N, 3) array of (latitude [deg], longitude [deg], altitude [m])
    """
    lla_array = []

    for r in traj_fixed_m[:, :3]:  # use only x, y, z
        x, y, z = r
        r_norm = np.linalg.norm([x, y, z])
        lat = np.degrees(np.arcsin(z / r_norm))
        lon = np.degrees(np.arctan2(y, x))
        alt = r_norm - R_moon
        lla_array.append([lat, lon, alt])

    return np.array(lla_array)

def mci_to_lla(x_inertial, y_inertial, z_inertial, t=0.0, theta0=0.0):
    # Convert MCI to Moon-fixed frame
    R_moon = 1.7374e6  # meters
    omega_moon = 2 * np.pi / (27.321661 * 86400)  # Moon's angular velocity (rad/s)
    
    # Calculate rotation angle
    theta = theta0 + omega_moon * t
    
    # Create inverse rotation matrix (transpose of original)
    R_z_inv = np.array([
        [np.cos(theta), np.sin(theta), 0],
        [-np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    
    # Rotate from inertial to fixed frame
    r_inertial = np.array([x_inertial, y_inertial, z_inertial])
    r_fixed = R_z_inv @ r_inertial
    x_fixed, y_fixed, z_fixed = r_fixed
    
    # Convert fixed Cartesian to LLA
    xy_fixed = np.sqrt(x_fixed**2 + y_fixed**2)
    
    # Calculate longitude (handle poles)
    lon_rad = np.arctan2(y_fixed, x_fixed) if xy_fixed > 1e-9 else 0.0
    
    # Calculate latitude
    lat_rad = np.arctan2(z_fixed, xy_fixed)
    
    # Calculate altitude
    alt_m = np.linalg.norm(r_fixed) - R_moon
    
    return np.degrees(lat_rad), np.degrees(lon_rad), alt_m

def get_enu_to_mci_rotation(lat_rad, lon_rad):
    slon = np.sin(lon_rad)
    clon = np.cos(lon_rad)
    slat = np.sin(lat_rad)
    clat = np.cos(lat_rad)

    return np.array([
        [-slon,          -clon*slat,     clon*clat],
        [ clon,          -slon*slat,     slon*clat],
        [ 0.0,            clat,          slat     ]
    ])