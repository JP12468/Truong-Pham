# Re-importing libraries and re-defining the simulation after state reset
import numpy as np
import matplotlib.pyplot as plt

# Define parameters
L = 10  # Length of the domain
z0 = 1  # Position of the real source
t0 = 0  # Initial time
D = 1   # Diffusion coefficient
time = 1.0  # Time at which we evaluate the solution

# Free-space Green's function
def G_free(t, z, z_n, D):
    if t <= 0:
        return 0
    return (1 / np.sqrt(4 * np.pi * D * t)) * np.exp(-((z - z_n) ** 2) / (4 * D * t))

# Compute the total solution using the method of images
# Correcting the issue with (-1)^n when n is negative
# Using an alternative approach for alternating sign

def psi_total(t, z, z0, L, D):
    total = 0
    n_values = np.arange(-10, 11)  # Summing over n from -10 to 10 for convergence
    for n in n_values:
        z_n = (2 * n + 1) * L - z0  # Position of the image source
        sign = -1 if n % 2 != 0 else 1  # Alternate sign without using (-1)^n directly
        total += sign * G_free(t, z, z_n, D)
    return total

# Recompute the total solution
z_values = np.linspace(0, L, 500)
t_values = np.linspace(0.1, 10, 500)
psi_values = [psi_total(t_values, z, z0, L, D)   for z in z_values]

# Plot the solution
plt.figure(figsize=(10, 6))
plt.plot(z_values, psi_values, label=f"Solution at t={time}s", color="blue")
plt.axvline(z0, color="red", linestyle="--", label="Real Source (z0)")
plt.axvline(L, color="black", linestyle="--", label="Boundary (z=L)")
plt.axvline(0, color="black", linestyle="--", label="Boundary (z=0)")
plt.title("Solution Using Method of Images")
plt.xlabel("z (Position)")
plt.ylabel("ψ(t, z)")
plt.legend()
plt.grid()
plt.show()
