# Deterministic Gaussian Sampling

Deterministic approximation and reduction of multivariate **Dirac mixtures** and **Gaussian distributions** using an optimized C++ backend with a clean Python interface.

The computational core is written in C++ for high performance.  
The Python package provides a NumPy-friendly API and ships with **precompiled binaries**.

---

## Installation

```bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ deterministic-gaussian-sampling
```

### Requirements

- Python ≥ 3.8  
- NumPy  

Optional (for visualization examples):

- SciPy  
- Matplotlib  

---

# Overview

The package provides two main classes:

```python
DiracToDiracApproximation
GaussianToDiracApproximation
```

They allow you to:

- Reduce large discrete sample sets to fewer deterministic points
- Approximate Gaussian distributions with optimized Dirac support points
- Compute the modified van Mises distance
- Compute the analytic gradient of the distance

---

# 1️⃣ Dirac-to-Dirac Reduction

Reduce `M` discrete samples in ℝᴺ to `L < M` optimized deterministic samples.

---

## Basic Example

```python
import deterministic_gaussian_sampling
import numpy as np

# Generate example data
num_points = 3000
N = 2
L = 12

x = np.random.normal(0, 1, num_points)
y = np.random.normal(0, 1, num_points)
original_points = np.column_stack((x, y))

# Create approximation object
d2d = deterministic_gaussian_sampling.DiracToDiracApproximation()

# Allocate output array (L x N)
reduced = np.empty((L, N))

# Run reduction (multi-threaded version)
result = d2d.approximate_thread_double(
    original_points,  # input samples (M x N)
    num_points,       # M
    L,                # number of target points
    N,                # dimension
    reduced           # output buffer
)

print("Success:", result.result)
print("Reduced points:\n", reduced)

del d2d
```

---

## Available Methods

```python
approximate_double(...)          # single-threaded
approximate_thread_double(...)   # multi-threaded
approximate_function_double(...) # custom weight functions
```

---

# 2️⃣ Gaussian-to-Dirac Approximation

Approximate a multivariate Gaussian distribution with `L` deterministic Dirac points.

---

## Standard Normal Example

```python
import deterministic_gaussian_sampling
import numpy as np

N = 2
L = 12

g2d = deterministic_gaussian_sampling.GaussianToDiracApproximation()

approx = np.empty((L, N))

result = g2d.approximate_snd_double(
    L,
    N,
    approx
)

print("Success:", result.result)
print("Dirac points:\n", approx)

del g2d
```

---

## Full Covariance Example

```python
import numpy as np
import deterministic_gaussian_sampling

Sigma = np.array([[2.0, 0.5],
                  [0.5, 2.0]])

N = 2
L = 12

g2d = deterministic_gaussian_sampling.GaussianToDiracApproximation()

approx = np.empty((L, N))

result = g2d.approximate_double(
    Sigma,  # covariance matrix
    L,
    N,
    approx
)

print("Success:", result.result)
print("Dirac points:\n", approx)

del g2d
```

The covariance matrix is internally diagonalized and the optimized points are automatically transformed back into the original coordinate system.

---

# Distance and Gradient

You can evaluate approximation quality directly.

### Compute Distance

```python
distance = d2d.approximate_double.modified_van_mises_distance_sq(
    y, M, L, N, x
)
```

### Compute Analytic Gradient

```python
gradient = d2d.approximate_double.modified_van_mises_distance_sq_derivative(
    y, M, L, N, x
)
```

This allows integration into:

- Custom optimization routines  
- Gradient-based machine learning workflows  
- Differentiable programming setups  

---

# Custom Weight Functions (Advanced)

You can define position-dependent weights:

```python
def wX(x):
    return np.exp(-np.sum(x**2))

def wXD(x):
    return -2 * x * np.exp(-np.sum(x**2))
```

Use them with:

```python
d2d.approximate_function_double(
    y, M, L, N, x,
    wX=wX,
    wXD=wXD
)
```

---

# Notes

- All heavy computations run in optimized C++  
- Python layer is lightweight and NumPy-based  
- Precompiled binaries are included in the package  
- Works on Linux and Windows  

---
