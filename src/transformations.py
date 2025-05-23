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

# Convert Moon centered moon fixed coordinates to latitude, longitude, and altitude (LLA)
def mcmf2lla(MCMF):
    R_moon=1737.4  # km
    # Assumes spherical Moon (selenocentric coordinates)

    r_x, r_y, r_z = MCMF

    # Compute radial distance from center
    r_norm = np.sqrt(r_x**2 + r_y**2 + r_z**2)

    # Longitude λ (degrees)
    lon_deg = np.degrees(np.arctan2(r_y, r_x))

    # Geocentric latitude ψ (degrees)
    lat_deg = np.degrees(np.arcsin(r_z / r_norm))

    # Altitude above lunar surface (assumes spherical Moon)
    altitude_km = r_norm - R_moon

    return lat_deg, lon_deg, altitude_km

def lla2mcmf(lon_deg, lat_deg, alt_km):
    R_moon=1737.4  # km
    
    # Convert angles to radians
    lat_rad = np.deg2rad(lat_deg)
    lon_rad = np.deg2rad(lon_deg)

    # Compute radial distance from Moon's center
    r = R_moon + alt_km

    # Convert to Cartesian coordinates
    x = r * np.cos(lat_rad) * np.cos(lon_rad)
    y = r * np.cos(lat_rad) * np.sin(lon_rad)
    z = r * np.sin(lat_rad)

    return np.array([x, y, z])

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
    traj_3d = []
    for x, y in trajectory_2d:
        dx = x - offset_x
        dy = y - offset_y
        pos_vec = landing_vec + dx * north + dy * up
        traj_3d.append(pos_vec)

    return np.array(traj_3d), landing_vec