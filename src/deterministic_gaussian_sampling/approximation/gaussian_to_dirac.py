import deterministic_gaussian_sampling.type_wrapper.ctypes_wrapper as ctypes_wrapper
import deterministic_gaussian_sampling.type_wrapper.python_variant as python_variant
import ctypes
import numpy
from typing import Optional
from .base_approximation import BaseApproximation


class GaussianToDiracApproximation(BaseApproximation):
    def __init__(self):
        super().__init__()
        cdll = self.__class__.cdll
        if cdll is None:
            raise OSError("C++-Library was not loaded. Unable to continue!!!")
        self.gm_to_dirac_double = cdll.create_gm_to_dirac_short_double()
        self.gm_to_dirac_snd_double = (
            cdll.create_gm_to_dirac_short_standard_normal_deviation_double()
        )

    def __del__(self):
        cdll = self.__class__.cdll
        if cdll is None:
            return
        cdll.delete_gm_to_dirac_short_double(self.gm_to_dirac_double)
        cdll.delete_gm_to_dirac_short_standard_normal_deviation_double(
            self.gm_to_dirac_snd_double
        )

    def approximate_double(
        self,
        covDiag: numpy.ndarray,
        L: int,
        N: int,
        x: numpy.ndarray,
        wX: Optional[numpy.ndarray] = None,
        options: Optional[python_variant.ApproximateOptionsPy] = None,
    ) -> python_variant.ApproximationResultPy:
        cdll = self.__class__.cdll
        if cdll is None:
            raise OSError("C++-Library was not loaded. Unable to continue!!!")
        minimizer_result = ctypes_wrapper.GslMinimizerResultCTypes()
        success = cdll.gm_to_dirac_short_double_approximate(
            self.gm_to_dirac_double,
            self._check_numpy_ndarray(covDiag, covDiag.shape[0], covDiag.shape[0]),
            ctypes.c_size_t(L),
            ctypes.c_size_t(N),
            ctypes.c_size_t(100),
            self._check_numpy_ndarray(x, L, N),
            self._check_numpy_ndarray(wX, L, 1),
            ctypes.byref(minimizer_result),
            (
                None
                if options is None
                else ctypes_wrapper.ApproximateOptionsCTypes.from_py_type(options)
            ),
        )
        return python_variant.ApproximationResultPy.from_ctypes(
            success, minimizer_result, x, L, N
        )

    def approximate_snd_double(
        self,
        L: int,
        N: int,
        x: numpy.ndarray,
        wX: Optional[numpy.ndarray] = None,
        options: Optional[python_variant.ApproximateOptionsPy] = None,
    ) -> python_variant.ApproximationResultPy:
        cdll = self.__class__.cdll
        if cdll is None:
            raise OSError("C++-Library was not loaded. Unable to continue!!!")
        minimizer_result = ctypes_wrapper.GslMinimizerResultCTypes()
        success = cdll.gm_to_dirac_short_standard_normal_deviation_double_approximate(
            self.gm_to_dirac_snd_double,
            ctypes.c_size_t(L),
            ctypes.c_size_t(N),
            ctypes.c_size_t(100),
            self._check_numpy_ndarray(x, L, N),
            self._check_numpy_ndarray(wX, L, 1),
            ctypes.byref(minimizer_result),
            (
                None
                if options is None
                else ctypes_wrapper.ApproximateOptionsCTypes.from_py_type(options)
            ),
        )
        return python_variant.ApproximationResultPy.from_ctypes(
            success, minimizer_result, x, L, N
        )
