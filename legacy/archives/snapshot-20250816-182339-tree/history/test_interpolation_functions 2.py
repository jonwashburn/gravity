#!/usr/bin/env python3
"""
Test different interpolation functions
======================================
The standard μ(x) = x/√(1+x²) might not be optimal.
Test alternatives.
"""

import numpy as np
import matplotlib.pyplot as plt

def mu_standard(x):
    """Standard MOND interpolation"""
    return x / np.sqrt(1 + x**2)

def mu_simple(x):
    """Simple interpolation"""
    return x / (1 + x)

def mu_exponential(x):
    """Exponential transition"""
    return 1 - np.exp(-x)

def mu_bekenstein(x, n=1):
    """Bekenstein's μ_n function"""
    return x / np.sqrt(1 + x**(2/n))

def mu_modified(x, alpha=0.5):
    """Modified with sharper transition"""
    return x / np.sqrt(1 + alpha * x**2)

# Plot comparison
x = np.logspace(-2, 2, 1000)

plt.figure(figsize=(10, 6))
plt.semilogx(x, mu_standard(x), 'b-', linewidth=2, label='Standard: x/√(1+x²)')
plt.semilogx(x, mu_simple(x), 'r--', linewidth=2, label='Simple: x/(1+x)')
plt.semilogx(x, mu_exponential(x), 'g-.', linewidth=2, label='Exponential: 1-exp(-x)')
plt.semilogx(x, mu_bekenstein(x, n=2), 'm:', linewidth=2, label='Bekenstein n=2')
plt.semilogx(x, mu_modified(x, alpha=2), 'c-', linewidth=2, label='Modified α=2')

plt.xlabel('x = a_N/a₀')
plt.ylabel('μ(x)')
plt.title('Interpolation Function Comparison')
plt.grid(True, alpha=0.3)
plt.legend()
plt.ylim(0, 1.1)

plt.savefig('interpolation_functions.png', dpi=150, bbox_inches='tight')
plt.show()

# Check asymptotic behavior
print("Asymptotic behavior:")
print("x << 1 (Deep MOND):")
for name, func in [('Standard', mu_standard), ('Simple', mu_simple), 
                   ('Modified α=2', lambda x: mu_modified(x, 2))]:
    x_small = 0.01
    print(f"  {name}: μ({x_small}) = {func(x_small):.4f} ≈ {x_small:.4f}x")

print("\nx >> 1 (Newtonian):")
for name, func in [('Standard', mu_standard), ('Simple', mu_simple), 
                   ('Modified α=2', lambda x: mu_modified(x, 2))]:
    x_large = 100
    print(f"  {name}: μ({x_large}) = {func(x_large):.4f} ≈ 1")

# The key insight: we need μ(x) that transitions MORE GRADUALLY
# This would reduce the enhancement in the intermediate regime where most galaxies live 