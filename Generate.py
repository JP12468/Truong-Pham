import numpy as np
import matplotlib.pyplot as plt

# Định nghĩa các tham số
kappa_E = 1
Q_E = 1
L = 10
x0 = 0
y0 = -1
z0 = 2
N_points = 1000  # Số điểm ngẫu nhiên

# Gieo ngẫu nhiên tọa độ (x, y) và thời gian t trong phạm vi hợp lý
x = np.random.uniform(-10, 10, N_points)
y = np.random.uniform(-10, 10, N_points)
t = np.random.uniform(0.01, 10, N_points)  # Tránh t quá nhỏ

# Tính psi(t, x, y, z=0)
psi = np.zeros(N_points)

# Xét một số giá trị của n trong khoảng hợp lý (-5 đến 5) để xấp xỉ tổng vô hạn
n_values = np.arange(-100, 100)

# Chuyển đổi (-1)^n thành kiểu float để tránh lỗi mũ số nguyên âm
psi = np.zeros(N_points, dtype=np.float64)

for n in n_values:
    zn = 2 * n * L + (-1.0)**n * z0  # Đảm bảo (-1)^n là float
    exponent = -((x - x0)**2 + (y - y0)**2 + (0 - zn)**2) / (4 * kappa_E * t)
    psi += ((-1.0)**n * Q_E / ( (4 * np.pi * kappa_E * t)**(3/2) )) * np.exp(exponent)

# Vẽ đồ thị psi theo x và y (màu thể hiện giá trị)
plt.figure(figsize=(8, 6))
plt.scatter(x, y, c=psi, cmap='viridis', s=10)
plt.colorbar(label=r'$\psi(t, x, y, z=0)$')
plt.xlabel("x")
plt.ylabel("y")
plt.title(r'$\psi(t, x, y, z=0)$ distribution')
plt.show()