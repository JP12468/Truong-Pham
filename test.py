import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Cần cho đồ thị 3D

# Tham số
t0 = 0      # Thời gian ban đầu
x0 = 0.5    # Chọn x0 = 0.5
y0 = 0.5    # Chọn y0 = 0.5, thỏa mãn x0 + y0 = 1
t_values = [0.01, 1, 5, 10]  # Các giá trị thời gian cần vẽ

# Tạo lưới giá trị cho x và y
x = np.linspace(-3, 3, 200)
y = np.linspace(-3, 3, 200)
X, Y = np.meshgrid(x, y)

# Tạo figure với 4 subplot (2 hàng x 2 cột)
fig = plt.figure(figsize=(16, 12))

for i, t in enumerate(t_values):
    psi = 1/np.sqrt(4 * np.pi * (t - t0)) * np.exp(-((X - x0)**2 + (Y - y0)**2) / (4 * (t - t0)))
    
    # Tạo subplot 3D
    ax = fig.add_subplot(2, 2, i + 1, projection='3d')
    surf = ax.plot_surface(X, Y, psi, cmap='viridis', edgecolor='none')
    ax.set_title(r"$\psi(x,y,t)$, $t = $" + f"{t}", fontsize=14)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_zlabel(r'$\psi(x,y,t)$', fontsize=12)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

plt.tight_layout()
plt.show()
