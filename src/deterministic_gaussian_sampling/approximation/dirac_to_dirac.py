import ctypes
import numpy
from typing import Optional
import deterministic_gaussian_sampling.type_wrapper.python_variant as python_variant
import deterministic_gaussian_sampling.type_wrapper.ctypes_wrapper as ctypes_wrapper
from .base_approximation import BaseApproximation


class DiracToDiracApproximation(BaseApproximation):
    def __init__(self):
        super().__init__()
        cdll = self.__class__.cdll
        if cdll is None:
            raise OSError("C++-Library was not loaded. Unable to continue!!!")
        self.d2d_short_double = cdll.create_dirac_to_dirac_approx_short_double()
        self.d2d_func_double = cdll.create_dirac_to_dirac_approx_short_function_double()
        self.d2d_thread_double = cdll.create_dirac_to_dirac_approx_short_thread_double()

    def __del__(self):
        cdll = self.__class__.cdll
        if cdll is None:
            return
        cdll.delete_dirac_to_dirac_approx_short_double(self.d2d_short_double)
        cdll.delete_dirac_to_dirac_approx_short_function_double(self.d2d_func_double)
        cdll.delete_dirac_to_dirac_approx_short_thread_double(self.d2d_thread_double)

    def approximate_double(
        self,
        y: numpy.ndarray,
        M: int,
        L: int,
        N: int,
        x: numpy.ndarray,
        wX: Optional[numpy.ndarray] = None,
        wY: Optional[numpy.ndarray] = None,
        options: Optional[python_variant.ApproximateOptionsPy] = None,
    ) -> python_variant.ApproximationResultPy:
        cdll = self.__class__.cdll
        if cdll is None:
            raise OSError("C++-Library was not loaded. Unable to continue!!!")
        minimizer_result = ctypes_wrapper.GslMinimizerResultCTypes()
        success: ctypes.c_bool = cdll.dirac_to_dirac_approx_short_double_approximate(
            self.d2d_short_double,
            self._check_numpy_ndarray(y, M, N),
            ctypes.c_size_t(M),
            ctypes.c_size_t(L),
            ctypes.c_size_t(N),
            ctypes.c_size_t(100),
            self._check_numpy_ndarray(x, L, N),
            self._check_numpy_ndarray(wX, L, 1),
            self._check_numpy_ndarray(wY, M, 1),
            ctypes.byref(minimizer_result),
            (
                None
                if options is None
                else ctypes.byref(
                    ctypes_wrapper.ApproximateOptionsCTypes.from_py_type(options)
                )
            ),
        )
        return python_variant.ApproximationResultPy.from_ctypes(
            success, minimizer_result, x, L, N
        )

    def approximate_function_double(
        self,
        y: numpy.ndarray,
        M: int,
        L: int,
        N: int,
        x: numpy.ndarray,
        wX: Optional[numpy.ndarray] = None,
        wY: Optional[numpy.ndarray] = None,
        options: Optional[python_variant.ApproximateOptionsPy] = None,
    ) -> python_variant.ApproximationResultPy:
        cdll = self.__class__.cdll
        if cdll is None:
            raise OSError("C++-Library was not loaded. Unable to continue!!!")
        minimizer_result = ctypes_wrapper.GslMinimizerResultCTypes()
        success = cdll.dirac_to_dirac_approx_short_function_double_approximate(
            self.d2d_func_double,
            self._check_numpy_ndarray(y, M, N),
            ctypes.c_size_t(M),
            ctypes.c_size_t(L),
            ctypes.c_size_t(N),
            ctypes.c_size_t(100),
            self._check_numpy_ndarray(x, L, N),
            wX,  # TODO: replace with function wrapper
            wY,  # TODO: replace with function wrapper
            ctypes.byref(minimizer_result),
            (
                None
                if options is None
                else ctypes.byref(
                    ctypes_wrapper.ApproximateOptionsCTypes.from_py_type(options)
                )
            ),
        )
        return python_variant.ApproximationResultPy.from_ctypes(
            success, minimizer_result, x, L, N
        )

    def approximate_thread_double(
        self,
        y: numpy.ndarray,
        M: int,
        L: int,
        N: int,
        x: numpy.ndarray,
        wX: Optional[numpy.ndarray] = None,
        wY: Optional[numpy.ndarray] = None,
        options: Optional[python_variant.ApproximateOptionsPy] = None,
    ) -> python_variant.ApproximationResultPy:
        cdll = self.__class__.cdll
        if cdll is None:
            raise OSError("C++-Library was not loaded. Unable to continue!!!")
        minimizer_result = ctypes_wrapper.GslMinimizerResultCTypes()
        success = cdll.dirac_to_dirac_approx_short_thread_double_approximate(
            self.d2d_thread_double,
            self._check_numpy_ndarray(y, M, N),
            ctypes.c_size_t(M),
            ctypes.c_size_t(L),
            ctypes.c_size_t(N),
            ctypes.c_size_t(100),
            self._check_numpy_ndarray(x, L, N),
            self._check_numpy_ndarray(wX, L, 1),
            self._check_numpy_ndarray(wY, M, 1),
            ctypes.byref(minimizer_result),
            (
                None
                if options is None
                else ctypes.byref(
                    ctypes_wrapper.ApproximateOptionsCTypes.from_py_type(options)
                )
            ),
        )
        return python_variant.ApproximationResultPy.from_ctypes(
            success, minimizer_result, x, L, N
        )
