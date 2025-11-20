import ctypes
import numpy

class DiracToDiracApproximation:
    def __init__(self, cdll: ctypes.CDLL):
        self.cdll = cdll
        self.d2d_short_double = cdll.create_dirac_to_dirac_approx_short_double()
        self.d2d_short_float = cdll.create_dirac_to_dirac_approx_short_float()
        self.d2d_func_double = cdll.create_dirac_to_dirac_approx_short_function_double()
        self.d2d_thread_double = cdll.create_dirac_to_dirac_approx_short_thread_double()
        self.d2d_thread_float = cdll.create_dirac_to_dirac_approx_short_thread_float()

    def __del__(self):
        self.cdll.delete_dirac_to_dirac_approx_short_double(self.d2d_short_double)
        self.cdll.delete_dirac_to_dirac_approx_short_float(self.d2d_short_float)
        self.cdll.delete_dirac_to_dirac_approx_short_function_double(self.d2d_func_double)
        self.cdll.delete_dirac_to_dirac_approx_short_thread_double(self.d2d_thread_double)
        self.cdll.delete_dirac_to_dirac_approx_short_thread_float(self.d2d_thread_float)

    def _map_ctypes_numpy(self, ctype) -> type:
        if ctype == ctypes.c_double:
            return numpy.float64
        elif ctype == ctypes.c_float:
            return numpy.float32
        elif ctype == ctypes.c_size_t:
            return numpy.uint64
        else:
            raise TypeError("Unsupported ctype for mapping to numpy dtype")

    def _check_numpy_array(self, arr: numpy.ndarray, dtype, size) -> ctypes.Array:
        if not isinstance(arr, numpy.ndarray):
            raise TypeError("Input must be a numpy array")
        if arr.dtype != self._map_ctypes_numpy(dtype):
            raise TypeError(f"Input array must be of type {dtype}")
        if arr.size != size:
            raise ValueError(f"Input array must have size {size}")
        return arr.ctypes.data_as(ctypes.POINTER(dtype))
    
    def _check_numpy_matrix(self, mat: numpy.matrix, dtype, rows, cols) -> ctypes.Array:
        if not isinstance(mat, numpy.matrix):
            raise TypeError("Input must be a numpy matrix")
        if mat.dtype != self._map_ctypes_numpy(dtype):
            raise TypeError(f"Input matrix must be of type {dtype}")
        if mat.shape != (rows, cols):
            raise ValueError(f"Input matrix must have shape ({rows}, {cols})")
        return mat.ctypes.data_as(ctypes.POINTER(dtype))

    def approximate_double(self, y: numpy.matrix, M: int, L: int, N: int, x: numpy.matrix, wX: numpy.ndarray, wY: numpy.ndarray, result, options) -> (bool, numpy.matrix):
        x = (ctypes.c_double * N)()
        success = self.cdll.dirac_to_dirac_approx_short_double_approximate(
            self.d2d_short_double,
            self._check_numpy_matrix(y, ctypes.c_double, M, L),
            ctypes.c_size_t(M),
            ctypes.c_size_t(L),
            ctypes.c_size_t(N),
            ctypes.c_size_t(100),
            self._check_numpy_matrix(x, ctypes.c_double, L, N),
            self._check_numpy_array(wX, ctypes.c_double, L),
            self._check_numpy_array(wY, ctypes.c_double, M),
            ctypes.byref(result),
            ctypes.byref(options)
        )
        return success, x
    
    # todo pass function for weights
    # todo returen propper python types

    def approximate_float(self, y, M, L, N, bMax, wX, wY, result, options) -> (bool, ctypes.Array):
        x = (ctypes.c_float * N)()
        success = self.cdll.dirac_to_dirac_approx_short_float_approximate(
            self.d2d_short_float,
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
        return success, x
    
    def approximate_function_double(self, y, M, L, N, bMax, wX, wY, result, options) -> (bool, ctypes.Array):
        x = (ctypes.c_double * N)()
        success = self.cdll.dirac_to_dirac_approx_short_function_double_approximate(
            self.d2d_func_double,
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
        return success, x
    
    def approximate_thread_double(self, y, M, L, N, bMax, wX, wY, result, options) -> (bool, ctypes.Array):
        x = (ctypes.c_double * N)()
        success = self.cdll.dirac_to_dirac_approx_short_thread_double_approximate(
            self.d2d_thread_double,
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
        return success, x
    
    def approximate_thread_float(self, y, M, L, N, bMax, wX, wY, result, options) -> (bool, ctypes.Array):
        x = (ctypes.c_float * N)()
        success = self.cdll.dirac_to_dirac_approx_short_thread_float_approximate(
            self.d2d_thread_float,
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
        return success, x
    