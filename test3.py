import matplotlib.pyplot as plt
import jax.random as jr
import jax.numpy as jnp
import numpy as np  # Needed for creating linspace for Gaussian PDF comparison

# Parameters
x_0_max = 2  # Maximum for uniform distribution
sigma = 0.5  # Standard deviation for Gaussian distribution
num_samples = 10000  # Number of samples

# Create a random key for JAX
key = jr.PRNGKey(42)  # Seed for reproducibility

# Part 1: Uniform distribution for x_0
key, subkey = jr.split(key)  # Split the random key
x_0_uniform = jr.uniform(subkey, shape=(num_samples,), minval=-x_0_max, maxval=x_0_max)

# Part 2: Gaussian distribution for x_0
key, subkey = jr.split(key)  # Split the random key
x_0 = jr.normal(subkey, shape=(num_samples,)) * sigma

# Plot the results
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Uniform distribution plot
axes[0].hist(x_0_uniform, bins=50, density=True, alpha=0.6, color='g', label=r"$g(x_0)$ for uniform distribution")
axes[0].axhline(1 / (2 * x_0_max), color='r', linestyle='dashed', label=r"Uniform PDF")
axes[0].set_title(r"Uniform Distribution of $x_0$ for $x_{0,max} = 2$")
axes[0].set_xlabel(r"$x_0$")
axes[0].set_ylabel("Density")
axes[0].legend()
axes[0].grid(True)
axes[0].set_xlim(-2,2)

# Gaussian distribution plot
axes[1].hist(x_0, bins=50, density=True, alpha=0.6, color='b', label=r"$g(x_0)$ for Gaussian distribution")
# Correct Gaussian PDF for comparison
x_vals = np.linspace(-2, 2, 100)
gaussian_pdf = (1 / (np.sqrt(2 * np.pi * sigma**2))) * np.exp(-x_vals**2 / (2 * sigma**2))
axes[1].plot(x_vals, gaussian_pdf, 'r', label=r"Gaussian PDF")
axes[1].set_title(r"Gaussian Distribution of $x_0$ for $\sigma = 0.5$")
axes[1].set_xlabel(r"$x_0$")
axes[1].set_ylabel("Density")
axes[1].legend()
axes[1].grid(True)

# Show the plots
plt.tight_layout()
plt.show()
