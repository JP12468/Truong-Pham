import jax.numpy as jnp
from jax import vmap
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np

# Các hằng số
kappa_E = 1
Q_E = 1
L = 10
x0 = 0
y0 = -1
z0 = 2
N_points = 1000

# Tạo số ngẫu nhiên với JAX
key = random.PRNGKey(0)
key_x, key_y, key_t = random.split(key, 3)

x = random.uniform(key_x, (N_points,), minval=-10, maxval=10)
y = random.uniform(key_y, (N_points,), minval=-10, maxval=10)
t = random.uniform(key_t, (N_points,), minval=0.01, maxval=10)

# Giá trị của n trong tổng
n_values = jnp.arange(-50, 50)  # Giảm bớt để tăng tính ổn định số

# Hàm tính giá trị psi cho từng điểm (t, x, y)
def psi_single_point(t, x, y):
    zn = 2 * n_values * L + (-1.0)**n_values * z0
    exponent = -((x - x0)**2 + (y - y0)**2 + (0 - zn)**2) / (4 * kappa_E * t)
    psi_n = ((-1.0)**n_values * Q_E / ((4 * jnp.pi * kappa_E * t)**(3/2))) * jnp.exp(exponent)
    return jnp.sum(psi_n)

# Vector hóa tính toán cho toàn bộ điểm
psi_vectorized = vmap(psi_single_point)
psi = psi_vectorized(t, x, y)

# Sắp xếp dữ liệu theo t để vẽ đường xu hướng
sorted_indices = jnp.argsort(t)
t_sorted = np.array(t[sorted_indices])  # Đổi sang numpy để fit đường cong dễ hơn
psi_sorted = np.array(psi[sorted_indices])

# Fit một đường đa thức bậc 3 để làm mịn dữ liệu
poly_coeffs = np.polyfit(t_sorted, psi_sorted, 3)  # Fit bậc 3
psi_trend = np.polyval(poly_coeffs, t_sorted)

# Vẽ hai biểu đồ trong cùng một khung hình
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Biểu đồ (x, y) với màu thể hiện psi
sc1 = axes[0].scatter(x, y, c=psi, cmap='viridis', s=10)
fig.colorbar(sc1, ax=axes[0], label=r'$\psi(t, x, y, z=0)$')
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].set_title(r'$\psi(t, x, y, z=0)$ distribution')

# Biểu đồ (t, psi) và đường mô phỏng
sc2 = axes[1].scatter(t, psi, c=t, cmap='plasma', s=10, alpha=0.5, label="Raw Data")
axes[1].plot(t_sorted, psi_trend, color='red', linewidth=2, label="Fitted Trend (Poly-3)")
fig.colorbar(sc2, ax=axes[1], label=r'Time $t$')
axes[1].set_xlabel("t")
axes[1].set_ylabel(r'$\psi(t, x, y, z=0)$')
axes[1].set_title(r'$\psi$ vs. Time $t$ with Trend Line')
axes[1].legend()

plt.tight_layout()
plt.show()