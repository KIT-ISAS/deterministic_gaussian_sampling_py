import ctypes
from ctypes import c_double
from dynamic_dll_loading import load_dll, ApproximateOptions, GslminimizerResult
import numpy as np

lib = load_dll()






# Usage
instance = lib.create_dirac_to_dirac_approx_short_double()
print("Created C++ object at address:", instance)



# Dimensions
M, L, N, bMax = 30, 5, 3, 100

# Input arrays
y = (c_double * (M * N))(*np.random.rand(M * N))
x = (c_double * (L * N))()                # Output array
wX = (c_double * L)(*np.ones(L))
wY = (c_double * M)(*np.ones(M))

# Structs
result = GslminimizerResult()
options = ApproximateOptions(
    xtolAbs=1e-8,
    xtolRel=1e-8,
    ftolAbs=0,
    ftolRel=0,
    gtol=1e-8,
    initialX=False,
    maxIterations=100000,
    verbose=False
)

# Call the function
success = lib.dirac_to_dirac_approx_short_double_approximate(
    instance,
    y,
    M,
    L,
    N,
    bMax,
    x,
    wX,
    wY,
    ctypes.byref(result),
    ctypes.byref(options)
)

# ----------------------
# Results
# ----------------------
print("Success:", success)
print("x:", list(x))
print("Result iterations:", result.iterations)

print(result)

# Delete the object when done
lib.delete_dirac_to_dirac_approx_short_double(instance)
print("Deleted C++ object")