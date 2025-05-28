import numpy as np
import matplotlib.pyplot as plt

def generate_trajectory(h0=15000, vx0=1600, vy0=0, dt=0.1, T_max=17000, m=2000):
    g = 1.62  # lunar gravity (m/s^2)
    t = 0
    h = h0
    vx = vx0
    vy = vy0
    x = 0

    traj = []

    while h > 0:
        speed_x = abs(vx)
        speed_y = abs(vy)

        # === Phase 1: Kill Horizontal Velocity ===
        if speed_x > 50:
            thrust_angle = np.arctan2(-vy + 2, -vx + 1e-6)
            thrust_mag = min(T_max, m * np.hypot(vx, vy + g))
            phase = 1

        # === Phase 2: Controlled Vertical Descent ===
        elif speed_y > 2 and h > 100:
            # Blend horizontal and vertical control
            vx_target = 0.0
            vy_target = -10.0

            vx_error = vx - vx_target
            vy_error = vy - vy_target

            # Adjust thrust direction: tilt slightly against vx
            thrust_angle = np.arctan2(-vy_error, -vx_error + 1e-3)

            # Magnitude: enough to counter both descent and drift
            ax_des = vx_error / dt
            ay_des = vy_error / dt + g
            acc_total = np.hypot(ax_des, ay_des)

            thrust_mag = np.clip(m * acc_total, 0, T_max)
            phase = 2

        # === Phase 3: Final Touchdown ===
        else:
            thrust_angle = np.pi / 2  # vertical
            target_vy = -0.5
            vy_error = vy - target_vy
            thrust_mag = np.clip(m * (abs(vy_error) / dt + g), 0, T_max)
            phase = 3

        # Apply dynamics
        ax = (thrust_mag / m) * np.cos(thrust_angle)
        ay = (thrust_mag / m) * np.sin(thrust_angle) - g

        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        h += vy * dt
        t += dt

        traj.append([t, x, h, vx, vy, thrust_mag, thrust_angle, phase])

        if h <= 0 and abs(vx) < 0.5 and abs(vy) < 1.0:
            break

    return np.array(traj)