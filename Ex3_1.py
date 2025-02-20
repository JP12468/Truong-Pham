import numpy as np
import matplotlib.pyplot as plt

# Number of random points
N = 1000
R = 1

# Generate random points in a circle using rejection sampling
theta = np.random.uniform(0, 2 * np.pi, N)  # Random angles
r = np.sqrt(np.random.uniform(0, 1, N)) * R  # Random radii (sqrt for uniform disk)

# Convert polar to Cartesian coordinates
x0 = r * np.cos(theta)
y0 = r * np.sin(theta)

# Create a circle outline
circle_theta = np.linspace(0, 2 * np.pi, 300)
circle_x = R * np.cos(circle_theta)
circle_y = R * np.sin(circle_theta)

# Plot the generated points and the circle
plt.figure(figsize=(12, 6))
plt.scatter(x0, y0, s=5, alpha=0.6, color="blue", label="Random points")
plt.plot(circle_x, circle_y, 'r-', linewidth=2, label="Boundary (R=1)")
plt.xlabel("$x_0$")
plt.ylabel("$y_0$")
plt.xlim(-R, R)
plt.ylim(-R, R)
plt.gca().set_aspect('equal', adjustable='box')
plt.legend()
plt.title("Randomly generated points inside a circle (R=1)")
plt.show()
