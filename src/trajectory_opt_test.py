import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds

# Constants
mu_moon = 4.9048695e12  # m^3/s^2
R_moon = 1737.4e3       # meters
mass_lander = 1500     # kg
max_thrust = 45000      # N

# Time discretization
N = 20  # number of time steps
s_dim = 4  # [x, y, vx, vy]
u_dim = 2  # [thrust, angle]

# Initial and final states
s_0 = np.array([0, 15000, 1600, 0])  # x, y, vx, vy
s_f = np.array([0, 0, 0, 0])

def unpack_decision_variables(z):
    t_f = z[0]
    s = z[1:1 + (N + 1) * s_dim].reshape((N + 1, s_dim))
    u = z[1 + (N + 1) * s_dim:].reshape((N, u_dim))
    return t_f, s, u

def pack_decision_variables(t_f, s, u):
    return np.concatenate([[t_f], s.flatten(), u.flatten()])

def lunar_cost(z):
    t_f, _, u = unpack_decision_variables(z)
    dt = t_f / N
    return np.sum(u[:, 0] * dt)  # minimize total thrust (proxy for fuel)

def lunar_eq_constraints(z):
    t_f, s, u = unpack_decision_variables(z)
    dt = t_f / N
    constraint_list = []

    for i in range(N):
        x, y, vx, vy = s[i]
        x_next, y_next, vx_next, vy_next = s[i + 1]
        thrust, angle = u[i]

        ax = (thrust / mass_lander) * np.cos(angle)
        ay = (thrust / mass_lander) * np.sin(angle) - mu_moon / (R_moon + y)**2

        constraint_list.append([(x_next - x)/dt - vx,
                                (y_next - y)/dt - vy,
                                (vx_next - vx)/dt - ax,
                                (vy_next - vy)/dt - ay])

    # Initial and final state constraints
    constraint_list.append(s[0] - s_0)
    constraint_list.append(s[-1] - s_f)

    return np.concatenate(constraint_list)

# Initial guess
z_guess = pack_decision_variables(
    1000.0,
    np.linspace(s_0, s_f, N + 1),
    np.tile(np.array([max_thrust / 2, 0]), (N, 1))
)

# Bounds
bounds = Bounds(
    pack_decision_variables(1.0, -np.inf * np.ones((N + 1, s_dim)), np.array([0.0, -np.pi]) * np.ones((N, u_dim))),
    pack_decision_variables(500.0, np.inf * np.ones((N + 1, s_dim)), np.array([max_thrust, np.pi]) * np.ones((N, u_dim)))
)

print("Initial constraint violation:", np.linalg.norm(lunar_eq_constraints(z_guess)))

# Optimize
result = minimize(
    lunar_cost,
    z_guess,
    bounds=bounds,
    constraints={'type': 'eq', 'fun': lunar_eq_constraints},
    options={'maxiter': 500, 'disp': True}
)

# Plot if successful
if result.success:
    t_f_opt, s_opt, u_opt = unpack_decision_variables(result.x)

    x_vals = s_opt[:, 0]
    y_vals = s_opt[:, 1]
    vx_vals = s_opt[:, 2]
    vy_vals = s_opt[:, 3]
    t_vals = np.linspace(0, t_f_opt, N + 1)

    # Trajectory plot
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(x_vals / 1e3, y_vals / 1e3, marker='o')
    plt.xlabel('Downrange Distance [km]')
    plt.ylabel('Altitude [km]')
    plt.title('Lunar Lander Trajectory')
    plt.grid(True)

    # Velocity plot
    vel_mag = np.sqrt(vx_vals**2 + vy_vals**2)
    plt.subplot(1, 2, 2)
    plt.plot(t_vals, vel_mag, label='|v|')
    plt.plot(t_vals, vx_vals, '--', label='vx')
    plt.plot(t_vals, vy_vals, '--', label='vy')
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity [m/s]')
    plt.title('Velocity Profiles')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
else:
    print("Optimization failed:", result.message)
