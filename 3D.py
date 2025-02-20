import numpy as np
import matplotlib.pyplot as plt

# Tham số
t0 = 0      
x0 = 1.0    
y0 = 1.0    
t_values = [0.01, 1, 5, 10]  # Các giá trị thời gian cần vẽ

# Tạo lưới giá trị cho x và y
x = np.linspace(-10, 10, 1000)
y = np.linspace(-10, 10, 1000)
X, Y = np.meshgrid(x, y)

# Tạo figure với 4 subplot dạng mặt phẳng (2 hàng x 2 cột)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, t in enumerate(t_values):
    # Tính hàm Green
    psi = 1/np.sqrt(4 * np.pi * (t - t0)) * np.exp(-((X - x0)**2 + (Y - y0)**2) / (4 * (t - t0)))
    
    # Vẽ contour plot
    cp = axes[i].contourf(X, Y, psi, levels=50, cmap='viridis')
    axes[i].set_title(f'$t = {t}$', fontsize=14)
    axes[i].set_xlabel('$x$', fontsize=12)
    axes[i].set_ylabel('$y$', fontsize=12)
    fig.colorbar(cp, ax=axes[i])
    
plt.suptitle(r"Biểu diễn hàm Green $\psi(x,y,t)$ trên mặt phẳng", fontsize=16, y=0.95)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()