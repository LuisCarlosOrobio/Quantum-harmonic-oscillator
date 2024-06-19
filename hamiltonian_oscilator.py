import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Constants and parameters
hbar = 1.0545718e-34  # Reduced Planck's constant (J s)
m = 9.10938356e-31    # Mass of the particle (kg)
omega = 1.0           # Angular frequency (rad/s)
N = 500               # Number of discretization points
L = 1e-9              # Size of the system (m)
T_max = 1e-15         # Maximum time (s)
dt = 1e-18            # Time step (s)

# Discretize space and time
x = np.linspace(-L, L, N)  # Spatial grid points
dx = x[1] - x[0]           # Spatial step size
time_steps = int(T_max / dt)  # Number of time steps
t = np.linspace(0, T_max, time_steps)  # Time grid points

# Construct the Hamiltonian matrix
T = - (hbar**2 / (2 * m * dx**2)) * (np.diag(np.ones(N-1), -1) - 2 * np.diag(np.ones(N), 0) + np.diag(np.ones(N-1), 1))
# Kinetic energy matrix as a tridiagonal matrix
V = 0.5 * m * omega**2 * np.diag(x**2)
# Potential energy matrix as a diagonal matrix
H = T + V
# Total Hamiltonian matrix

# Solve the eigenvalue problem
E, psi = np.linalg.eigh(H)
# Eigenvalues (E) and eigenvectors (psi) of the Hamiltonian

# Define the initial wave packet
x0 = 0
sigma = L / 10
psi_0 = np.exp(-(x - x0)**2 / (2 * sigma**2))  # Gaussian wave packet centered at x0
psi_0 /= np.sqrt(np.sum(psi_0**2) * dx)  # Normalize the wave packet

# Time evolution of the wave packet
def time_evolve(psi_0, E, psi, t):
    c_n = np.dot(psi.T, psi_0) * dx  # Expansion coefficients
    psi_t = np.sum(c_n[:, None] * psi * np.exp(-1j * E[:, None] * t / hbar), axis=0)  # Time-evolved wave packet
    return psi_t

# Create the animation
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create a sphere
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x_sphere = 0.05 * L * np.outer(np.cos(u), np.sin(v))  # X coordinates of the sphere
y_sphere = 0.05 * L * np.outer(np.sin(u), np.sin(v))  # Y coordinates of the sphere
z_sphere = 0.05 * L * np.outer(np.ones(np.size(u)), np.cos(v))  # Z coordinates of the sphere

# Set the plot limits initially
ax.set_xlim([-L, L])
ax.set_ylim([-L, L])
ax.set_zlim([-L, L])
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.set_zlabel('Z Position (m)')
ax.set_title('Quantum Harmonic Oscillator')

# Initialize the plot
particle = [ax.plot_surface(x_sphere, y_sphere, z_sphere, color='b')]

# Update function for the animation
def update(frame):
    global particle
    for p in particle:
        p.remove()  # Remove the previous particle position
    t_frame = frame * dt  # Current time
    psi_t = time_evolve(psi_0, E, psi, t_frame)  # Time-evolved wave packet
    probability_density = np.abs(psi_t)**2  # Probability density
    max_prob_index = np.argmax(probability_density)  # Index of the maximum probability density
    x_pos = x[max_prob_index]  # X position of the particle
    y_pos = 0  # Y position (kept 0 for 1D problem)
    z_pos = 0  # Z position (kept 0 for 1D problem)
    particle[0] = ax.plot_surface(x_sphere + x_pos, y_sphere + y_pos, z_sphere + z_pos, color='b')  # Update the particle position

# Create the animation
ani = FuncAnimation(fig, update, frames=time_steps, interval=50)
plt.show()