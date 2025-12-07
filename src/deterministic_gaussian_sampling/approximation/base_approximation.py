import ctypes
import numpy
from typing import Optional
from deterministic_gaussian_sampling.dll_handling import load_dll

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
        self, arr: numpy.ndarray | None, L: int, N: int
    ) -> ctypes.Array:
        if arr is None:
            return None
        if not isinstance(arr, numpy.ndarray):
            raise TypeError("Input must be a numpy array")
        if (
            arr.dtype != float
            and arr.dtype != numpy.float16
            and arr.dtype != numpy.float32
            and arr.dtype != numpy.float64
            and arr.dtype != numpy.float96
            and arr.dtype != numpy.float128
        ):
            raise TypeError(
                f"Input array must be of [float, numpy.float16, numpy.float32, numpy.float64, numpy.float96, numpy.float128], but got {arr.dtype}."
            )
        if arr.shape != (L, N):
            row, cols = arr.shape
            raise ValueError(
                f"Input array must have size [{L}x{N}] but got [{row}x{cols}]"
            )
        return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    @staticmethod
    def _register_cdll(cdll: ctypes.CDLL):
        BaseApproximation.cdll = cdll