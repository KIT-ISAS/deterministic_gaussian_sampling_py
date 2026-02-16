import ctypes
from dataclasses import dataclass
import numpy
from typing import Optional
from deterministic_gaussian_sampling.dll_handling import load_dll

@dataclass
class CovarianceData:
    cov: numpy.ndarray
    eigvals: numpy.ndarray
    Q: numpy.ndarray
    sqrt_eigvals: numpy.ndarray

    def __init__(self, cov: numpy.ndarray):
        self.cov = cov
        self.eigvals, self.Q = numpy.linalg.eigh(cov)
        self.eigvals = numpy.maximum(self.eigvals, 1e-14)
        self.sqrt_eigvals = numpy.sqrt(self.eigvals)

        self.cov = numpy.ascontiguousarray(self.cov, dtype=numpy.float64)
        self.Q = numpy.ascontiguousarray(self.Q, dtype=numpy.float64)
        self.eigvals = numpy.ascontiguousarray(self.eigvals, dtype=numpy.float64)
        self.sqrt_eigvals = numpy.ascontiguousarray(self.sqrt_eigvals, dtype=numpy.float64)

class BaseApproximation:
    cdll: Optional[ctypes.CDLL] = None

    def __init__(self):
        if self.cdll is None:
            lib = load_dll()
            self._register_cdll(lib)

    def _map_ctypes_numpy(self, ctype) -> type:
        if ctype == ctypes.c_double:
            return numpy.float64
        elif ctype == ctypes.c_float:
            return numpy.float32
        elif ctype == ctypes.c_size_t:
            return numpy.uint64
        else:
            raise TypeError("Unsupported ctype for mapping to numpy dtype")

    def _check_numpy_ndarray(
        self, arr: Optional[numpy.ndarray], L: int, N: Optional[int] = None
    ) -> Optional[numpy.ndarray]:
        if arr is None:
            return None
        if not isinstance(arr, numpy.ndarray):
            raise TypeError("Input must be a numpy array")
        if not numpy.issubdtype(arr.dtype, numpy.floating):
            raise TypeError(
                f"Input array must be of a floating type, but got {arr.dtype}."
            )
        if L <= 0 or (N is not None and N <= 0):
            raise ValueError("L and N must be positive integers.")
        arrShape = arr.shape
        if arr.ndim == 1:
            if arrShape[0] != L:
                raise ValueError(
                    f"Input array must have size [{L}] but got [{arrShape[0]}]"
                )
            if N is not None and N != 1:
                raise ValueError(
                    f"Input array must have size [{L}x{N}] but got [{L}]"
                )
        elif arr.ndim == 2:
            if arrShape != (L, N):
                row, cols = arrShape
                raise ValueError(
                    f"Input array must have size [{L}x{N}] but got [{row}x{cols}]"
                )
        else:
            raise ValueError("Input array must be 1D or 2D")

        # ensure contiguous float64 memory
        return numpy.ascontiguousarray(arr, dtype=numpy.float64)
    
    def _check_covariance_matrix(self, cov: numpy.ndarray, N: int, tol = 1e-6) -> CovarianceData:
        if cov.shape != (N, N):
            raise ValueError(f"Covariance matrix must be of shape [{N}x{N}] but got {cov.shape}")
        if not numpy.allclose(cov, cov.T, atol=tol):
            raise ValueError("Covariance matrix must be symmetric")
        if numpy.any(numpy.linalg.eigvalsh(cov) <= -tol):
            raise ValueError("Covariance matrix must be positive definite")
        
        covChecked = self._check_numpy_ndarray(cov, N, N)
        return CovarianceData(covChecked)
        
    def _check_weights(self, weight: Optional[numpy.ndarray], size: int, tol=1e-6) -> Optional[numpy.ndarray]:
        if weight is None:
            return None
        wChecked = self._check_numpy_ndarray(weight, size, None)
        if numpy.any(wChecked < 0):
            raise ValueError("Weights cannot be negative")
        if numpy.abs(1 - numpy.sum(wChecked)) > tol:
            raise ValueError(f"Sum of weights must be 1 within tolerance +/-{tol}, but got {numpy.sum(wChecked)}")
        return wChecked

    @staticmethod
    def _register_cdll(cdll: ctypes.CDLL):
        BaseApproximation.cdll = cdll
