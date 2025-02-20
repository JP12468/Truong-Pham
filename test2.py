import numpy as np
import matplotlib.pyplot as plt

L = 10  
x0 = 1  
x_vals = np.linspace(-10, 10, 1000)  

def psi(t, x, L, x0):
    psi_values = np.zeros_like(x)  
    
    for n in range(-1000, 1000):  
        x_n = 2 * L * n + (-1) ** n * x0
        psi_values += (-1) ** n / np.sqrt(4 * np.pi * t) * np.exp(-((x - x_n) ** 2) / (4 * t))
    
    return psi_values


t_values = [1, 5, 10]

plt.figure(figsize=(12, 8))

for t in t_values:
    psi_values = psi(t, x_vals, L, x0)
    plt.plot(x_vals, psi_values, label=f'$t={t}$')

plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.title(r'Solution $\psi(t, x)$ with Boundary Condition $\psi(t, x=L)=0$', fontsize=14)
plt.xlabel(r'$x$', fontsize=12)
plt.ylabel(r'$\psi(t, x)$', fontsize=12)
plt.legend()
plt.grid()
plt.show()