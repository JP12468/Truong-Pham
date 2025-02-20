import numpy as np
import matplotlib.pyplot as plt

# Constants
t0 = 0  # initial time
x0 = 1  # initial position
times = [0.1, 0.5, 1.0]  # Times to visualize
x = np.linspace(-5, 5, 200)  # x values for plotting

# Function to compute Green's function in 1D
def greens_function(t, x, t0, x0):
    if t <= t0:
        return np.zeros_like(x)
    return (1 / np.sqrt(4 * np.pi * (t - t0))) * np.exp(-((x - x0)**2) / (4 * (t - t0)))

# Plotting
plt.figure(figsize=(8, 6))

for t in times:
    Z = greens_function(t, x, t0, x0)
    plt.plot(x, Z, label=f't = {t}')

plt.title('Green\'s Function at Different Times')
plt.xlabel('$x$')
plt.ylabel('$\psi(t, x)$')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
    