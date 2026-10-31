import deterministic_gaussian_sampling.type_wrapper.ctypes_wrapper as ctypes_wrapper
import deterministic_gaussian_sampling.type_wrapper.python_variant as python_variant
import ctypes
import numpy
from typing import Optional
from .base_approximation import BaseApproximation

class _ApproximateDouble:
    """
    Gaussian-to-Dirac approximation with covariance matrix.

    The covariance matrix is internally diagonalized.
    The optimized points are transformed back to
    the original coordinate system.
    """
    def __init__(self, parent):
        self._parent = parent

    def __call__(
        self,
        cov: numpy.ndarray,
        L: int,
        N: int,
        x: numpy.ndarray,
        wX: Optional[numpy.ndarray] = None,
        options: Optional[python_variant.ApproximateOptionsPy] = None,
    ) -> python_variant.ApproximationResultPy:
        """
        Approximate Gaussian distribution by L Dirac points.

        Parameters
        ----------
        cov : numpy.ndarray
            Covariance matrix (N x N).
        L : int
            Number of Dirac components.
        N : int
            Dimension.
        x : numpy.ndarray
            Initial guess (L x N).
        wX : numpy.ndarray, optional
            Dirac weights.

        Returns
        -------
        ApproximationResultPy
            Result containing optimized Dirac points.
        """
        cdll = self._parent.cdll
        if cdll is None:
            raise OSError("C++-Library was not loaded. Unable to continue!!!")
        covData = self._parent._check_covariance_matrix(cov, N)
        xChecked = self._parent._check_numpy_ndarray(x, L, N)
        wXChecked = self._parent._check_weights(wX, L)
        minimizer_result = ctypes_wrapper.GslMinimizerResultCTypes()
        success = cdll.gm_to_dirac_short_double_approximate(
            self._parent.gm_to_dirac_double,
            covData.sqrt_eigvals.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_size_t(L),
            ctypes.c_size_t(N),
            xChecked.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            None if wXChecked is None else wXChecked.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(minimizer_result),
            (
                None
                if options is None
                else ctypes_wrapper.ApproximateOptionsCTypes.from_py_type(options)
            ),
        )
        result = python_variant.ApproximationResultPy.from_ctypes(
            success, minimizer_result, x, L, N
        )
        result.x = result.x @ covData.Q.T
        return result
    
    def modified_van_mises_distance_sq(
        self,
        cov: numpy.ndarray,
        L: int,
        N: int,
        x: numpy.ndarray,
        wX: Optional[numpy.ndarray] = None,
        bMax: int = 100
    ) -> float:
        cdll = self._parent.cdll
        if cdll is None:
            raise OSError("C++-Library was not loaded. Unable to continue!!!")
        covData = self._parent._check_covariance_matrix(cov, N)
        xChecked = self._parent._check_numpy_ndarray(x, L, N)
        wXChecked = self._parent._check_weights(wX, L)
        result = ctypes.c_double(0.0)
        cdll.gm_to_dirac_short_double_modified_van_mises_distance_sq(
            self._parent.gm_to_dirac_double,
            covData.sqrt_eigvals.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(result),
            ctypes.c_size_t(L),
            ctypes.c_size_t(N),
            ctypes.c_size_t(bMax),
            xChecked.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            None if wXChecked is None else wXChecked.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )
        return result.value
    
    def modified_van_mises_distance_sq_derivative(
        self,
        cov: numpy.ndarray,
        L: int,
        N: int,
        x: numpy.ndarray,
        wX: Optional[numpy.ndarray] = None,
        bMax: int = 100
    ) -> float:
        cdll = self._parent.cdll
        if cdll is None:
            raise OSError("C++-Library was not loaded. Unable to continue!!!")
        covData = self._parent._check_covariance_matrix(cov, N)
        xChecked = self._parent._check_numpy_ndarray(x, L, N)
        wXChecked = self._parent._check_weights(wX, L)
        gradient = numpy.zeros((L, N))
        gradientChecked = self._parent._check_numpy_ndarray(gradient, L, N)
        cdll.gm_to_dirac_short_double_modified_van_mises_distance_sq_derivative(
            self._parent.gm_to_dirac_double,
            covData.sqrt_eigvals.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            gradientChecked.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_size_t(L),
            ctypes.c_size_t(N),
            ctypes.c_size_t(bMax),
            xChecked.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            None if wXChecked is None else wXChecked.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )
        return gradient

class _ApproximateSNDDouble:
    """
    Gaussian-to-Dirac approximation for standard normal deviation case.

    Assumes standardized Gaussian structure.
    """
    def __init__(self, parent):
        self._parent = parent

    def __call__(
        self,
        L: int,
        N: int,
        x: numpy.ndarray,
        wX: Optional[numpy.ndarray] = None,
        options: Optional[python_variant.ApproximateOptionsPy] = None,
    ) -> python_variant.ApproximationResultPy:
        """
        Approximate standard normal Gaussian with L Dirac points.

        Parameters
        ----------
        L : int
            Number of Dirac components.
        N : int
            Dimension.
        x : numpy.ndarray
            Initial guess (L x N).

        Returns
        -------
        ApproximationResultPy
            Optimization result.
        """
        cdll = self._parent.cdll
        if cdll is None:
            raise OSError("C++-Library was not loaded. Unable to continue!!!")
        xChecked = self._parent._check_numpy_ndarray(x, L, N)
        wXChecked = self._parent._check_weights(wX, L)
        minimizer_result = ctypes_wrapper.GslMinimizerResultCTypes()
        success = cdll.gm_to_dirac_short_standard_normal_deviation_double_approximate(
            self._parent.gm_to_dirac_snd_double,
            ctypes.c_size_t(L),
            ctypes.c_size_t(N),
            xChecked.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            None if wXChecked is None else wXChecked.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
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
    
    def modified_van_mises_distance_sq(
        self,
        L: int,
        N: int,
        x: numpy.ndarray,
        wX: Optional[numpy.ndarray] = None,
        bMax: int = 100
    ) -> float:
        cdll = self._parent.cdll
        if cdll is None:
            raise OSError("C++-Library was not loaded. Unable to continue!!!")
        xChecked = self._parent._check_numpy_ndarray(x, L, N)
        wXChecked = self._parent._check_weights(wX, L)
        result = ctypes.c_double(0.0)
        cdll.gm_to_dirac_short_standard_normal_deviation_double_modified_van_mises_distance_sq(
            self._parent.gm_to_dirac_snd_double,
            ctypes.byref(result),
            ctypes.c_size_t(L),
            ctypes.c_size_t(N),
            ctypes.c_size_t(bMax),
            xChecked.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            None if wXChecked is None else wXChecked.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )
        return result.value
    
    def modified_van_mises_distance_sq_derivative(
        self,
        L: int,
        N: int,
        x: numpy.ndarray,
        wX: Optional[numpy.ndarray] = None,
        bMax: int = 100
    ) -> numpy.ndarray:
        cdll = self._parent.cdll
        if cdll is None:
            raise OSError("C++-Library was not loaded. Unable to continue!!!")
        xChecked = self._parent._check_numpy_ndarray(x, L, N)
        wXChecked = self._parent._check_weights(wX, L)
        gradient = numpy.zeros((L, N))
        gradientChecked = self._parent._check_numpy_ndarray(gradient, L, N)
        cdll.gm_to_dirac_short_standard_normal_deviation_double_modified_van_mises_distance_sq_derivative(
            self._parent.gm_to_dirac_snd_double,
            gradientChecked.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_size_t(L),
            ctypes.c_size_t(N),
            ctypes.c_size_t(bMax),
            xChecked.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            None if wXChecked is None else wXChecked.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )
        return gradient

class _ApproximateEvenDouble:
    """
    Closed-form Gaussian-to-Dirac approximation for even dimensions.

    Standard normal target without quadrature; isotropic covariance supported.
    """
    def __init__(self, parent):
        self._parent = parent

    def __call__(
        self,
        L: int,
        N: int,
        bMax: float,
        x: numpy.ndarray,
        wX: Optional[numpy.ndarray] = None,
        options: Optional[python_variant.ApproximateOptionsPy] = None,
    ) -> python_variant.ApproximationResultPy:
        """
        Approximate standard normal Gaussian with L Dirac points.

        Parameters
        ----------
        L : int
            Number of Dirac components.
        N : int
            Dimension (even, >= 2).
        bMax : float
            Upper bound of the b-integral (> 0).
        x : numpy.ndarray
            Initial guess (L x N).

        Returns
        -------
        ApproximationResultPy
            Optimization result.
        """
        cdll = self._parent.cdll
        if cdll is None:
            raise OSError("C++-Library was not loaded. Unable to continue!!!")
        self._parent._check_even_dimension(N)
        self._parent._check_positive(bMax, "bMax")
        xChecked = self._parent._check_numpy_ndarray(x, L, N)
        wXChecked = self._parent._check_weights(wX, L)
        minimizer_result = ctypes_wrapper.GslMinimizerResultCTypes()
        success = cdll.gm_to_dirac_even_closed_form_double_approximate(
            self._parent.gm_to_dirac_even_double,
            ctypes.c_size_t(L),
            ctypes.c_size_t(N),
            ctypes.c_double(bMax),
            xChecked.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            None if wXChecked is None else wXChecked.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(minimizer_result),
            (
                None
                if options is None
                else ctypes_wrapper.ApproximateOptionsCTypes.from_py_type(options)
            ),
        )
        return python_variant.ApproximationResultPy.from_ctypes(
            success, minimizer_result, xChecked, L, N
        )

    def approximate_isotropic(
        self,
        L: int,
        N: int,
        sigma: float,
        bMax: float,
        x: numpy.ndarray,
        wX: Optional[numpy.ndarray] = None,
        options: Optional[python_variant.ApproximateOptionsPy] = None,
    ) -> python_variant.ApproximationResultPy:
        """
        Approximate isotropic Gaussian N(0, sigma^2 I) with L Dirac points.

        Parameters
        ----------
        L : int
            Number of Dirac components.
        N : int
            Dimension (even, >= 2).
        sigma : float
            Standard deviation (> 0).
        bMax : float
            Upper bound of the b-integral (> 0).
        x : numpy.ndarray
            Initial guess (L x N).

        Returns
        -------
        ApproximationResultPy
            Optimization result.
        """
        cdll = self._parent.cdll
        if cdll is None:
            raise OSError("C++-Library was not loaded. Unable to continue!!!")
        self._parent._check_even_dimension(N)
        self._parent._check_positive(sigma, "sigma")
        self._parent._check_positive(bMax, "bMax")
        xChecked = self._parent._check_numpy_ndarray(x, L, N)
        wXChecked = self._parent._check_weights(wX, L)
        minimizer_result = ctypes_wrapper.GslMinimizerResultCTypes()
        success = cdll.gm_to_dirac_even_closed_form_double_approximate_isotropic(
            self._parent.gm_to_dirac_even_double,
            ctypes.c_size_t(L),
            ctypes.c_size_t(N),
            ctypes.c_double(sigma),
            ctypes.c_double(bMax),
            xChecked.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            None if wXChecked is None else wXChecked.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(minimizer_result),
            (
                None
                if options is None
                else ctypes_wrapper.ApproximateOptionsCTypes.from_py_type(options)
            ),
        )
        return python_variant.ApproximationResultPy.from_ctypes(
            success, minimizer_result, xChecked, L, N
        )

    def modified_van_mises_distance_sq(
        self,
        L: int,
        N: int,
        bMax: float,
        x: numpy.ndarray,
        wX: Optional[numpy.ndarray] = None
    ) -> float:
        cdll = self._parent.cdll
        if cdll is None:
            raise OSError("C++-Library was not loaded. Unable to continue!!!")
        self._parent._check_even_dimension(N)
        self._parent._check_positive(bMax, "bMax")
        xChecked = self._parent._check_numpy_ndarray(x, L, N)
        wXChecked = self._parent._check_weights(wX, L)
        result = ctypes.c_double(0.0)
        cdll.gm_to_dirac_even_closed_form_double_modified_van_mises_distance_sq(
            self._parent.gm_to_dirac_even_double,
            ctypes.byref(result),
            ctypes.c_size_t(L),
            ctypes.c_size_t(N),
            ctypes.c_double(bMax),
            xChecked.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            None if wXChecked is None else wXChecked.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )
        return result.value

    def modified_van_mises_distance_sq_derivative(
        self,
        L: int,
        N: int,
        bMax: float,
        x: numpy.ndarray,
        wX: Optional[numpy.ndarray] = None
    ) -> numpy.ndarray:
        cdll = self._parent.cdll
        if cdll is None:
            raise OSError("C++-Library was not loaded. Unable to continue!!!")
        self._parent._check_even_dimension(N)
        self._parent._check_positive(bMax, "bMax")
        xChecked = self._parent._check_numpy_ndarray(x, L, N)
        wXChecked = self._parent._check_weights(wX, L)
        gradient = numpy.zeros((L, N))
        gradientChecked = self._parent._check_numpy_ndarray(gradient, L, N)
        cdll.gm_to_dirac_even_closed_form_double_modified_van_mises_distance_sq_derivative(
            self._parent.gm_to_dirac_even_double,
            gradientChecked.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_size_t(L),
            ctypes.c_size_t(N),
            ctypes.c_double(bMax),
            xChecked.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            None if wXChecked is None else wXChecked.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )
        return gradient

    def modified_van_mises_distance_sq_isotropic(
        self,
        L: int,
        N: int,
        sigma: float,
        bMax: float,
        x: numpy.ndarray,
        wX: Optional[numpy.ndarray] = None
    ) -> float:
        cdll = self._parent.cdll
        if cdll is None:
            raise OSError("C++-Library was not loaded. Unable to continue!!!")
        self._parent._check_even_dimension(N)
        self._parent._check_positive(sigma, "sigma")
        self._parent._check_positive(bMax, "bMax")
        xChecked = self._parent._check_numpy_ndarray(x, L, N)
        wXChecked = self._parent._check_weights(wX, L)
        result = ctypes.c_double(0.0)
        cdll.gm_to_dirac_even_closed_form_double_modified_van_mises_distance_sq_isotropic(
            self._parent.gm_to_dirac_even_double,
            ctypes.byref(result),
            ctypes.c_size_t(L),
            ctypes.c_size_t(N),
            ctypes.c_double(sigma),
            ctypes.c_double(bMax),
            xChecked.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            None if wXChecked is None else wXChecked.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )
        return result.value

class _ApproximateEvenBinfDouble:
    """
    bMax-free closed-form Gaussian-to-Dirac approximation for even dimensions.

    Standard normal target; returned points have zero mean.
    """
    def __init__(self, parent):
        self._parent = parent

    def __call__(
        self,
        L: int,
        N: int,
        x: numpy.ndarray,
        wX: Optional[numpy.ndarray] = None,
        options: Optional[python_variant.ApproximateOptionsPy] = None,
    ) -> python_variant.ApproximationResultPy:
        """
        Approximate standard normal Gaussian with L Dirac points.

        Parameters
        ----------
        L : int
            Number of Dirac components.
        N : int
            Dimension (even, >= 2).
        x : numpy.ndarray
            Initial guess (L x N).

        Returns
        -------
        ApproximationResultPy
            Optimization result.
        """
        cdll = self._parent.cdll
        if cdll is None:
            raise OSError("C++-Library was not loaded. Unable to continue!!!")
        self._parent._check_even_dimension(N)
        xChecked = self._parent._check_numpy_ndarray(x, L, N)
        wXChecked = self._parent._check_weights(wX, L)
        minimizer_result = ctypes_wrapper.GslMinimizerResultCTypes()
        success = cdll.gm_to_dirac_even_binf_double_approximate(
            self._parent.gm_to_dirac_even_binf_double,
            ctypes.c_size_t(L),
            ctypes.c_size_t(N),
            xChecked.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            None if wXChecked is None else wXChecked.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(minimizer_result),
            (
                None
                if options is None
                else ctypes_wrapper.ApproximateOptionsCTypes.from_py_type(options)
            ),
        )
        return python_variant.ApproximationResultPy.from_ctypes(
            success, minimizer_result, xChecked, L, N
        )

    def modified_van_mises_distance_sq(
        self,
        L: int,
        N: int,
        x: numpy.ndarray,
        wX: Optional[numpy.ndarray] = None
    ) -> float:
        """Distance of the mean-corrected input."""
        cdll = self._parent.cdll
        if cdll is None:
            raise OSError("C++-Library was not loaded. Unable to continue!!!")
        self._parent._check_even_dimension(N)
        xChecked = self._parent._check_numpy_ndarray(x, L, N)
        wXChecked = self._parent._check_weights(wX, L)
        result = ctypes.c_double(0.0)
        cdll.gm_to_dirac_even_binf_double_modified_van_mises_distance_sq(
            self._parent.gm_to_dirac_even_binf_double,
            ctypes.byref(result),
            ctypes.c_size_t(L),
            ctypes.c_size_t(N),
            xChecked.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            None if wXChecked is None else wXChecked.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )
        return result.value

    def modified_van_mises_distance_sq_derivative(
        self,
        L: int,
        N: int,
        x: numpy.ndarray,
        wX: Optional[numpy.ndarray] = None
    ) -> numpy.ndarray:
        """Gradient w.r.t. the samples of the mean-corrected input."""
        cdll = self._parent.cdll
        if cdll is None:
            raise OSError("C++-Library was not loaded. Unable to continue!!!")
        self._parent._check_even_dimension(N)
        xChecked = self._parent._check_numpy_ndarray(x, L, N)
        wXChecked = self._parent._check_weights(wX, L)
        gradient = numpy.zeros((L, N))
        gradientChecked = self._parent._check_numpy_ndarray(gradient, L, N)
        cdll.gm_to_dirac_even_binf_double_modified_van_mises_distance_sq_derivative(
            self._parent.gm_to_dirac_even_binf_double,
            gradientChecked.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_size_t(L),
            ctypes.c_size_t(N),
            xChecked.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            None if wXChecked is None else wXChecked.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )
        return gradient

class GaussianToDiracApproximation(BaseApproximation):
    """
    Gaussian-to-Dirac approximation interface.

    Python wrapper around the C++ Gaussian approximation backend.

    Provides:

    - Diagonal covariance Gaussian approximation
    - Standard normal deviation variant
    - Closed-form even-dimensional variants (finite bMax and bMax-free)
    - Distance evaluation
    - Analytical gradient computation
    """
    def __init__(self):
        super().__init__()
        cdll = self.__class__.cdll
        if cdll is None:
            raise OSError("C++-Library was not loaded. Unable to continue!!!")
        self.gm_to_dirac_double = cdll.create_gm_to_dirac_short_double()
        self.gm_to_dirac_snd_double = (
            cdll.create_gm_to_dirac_short_standard_normal_deviation_double()
        )
        self.gm_to_dirac_even_double = cdll.create_gm_to_dirac_even_closed_form_double()
        self.gm_to_dirac_even_binf_double = cdll.create_gm_to_dirac_even_binf_double()

        self.approximate_double = _ApproximateDouble(self)
        self.approximate_snd_double = _ApproximateSNDDouble(self)
        self.approximate_even_double = _ApproximateEvenDouble(self)
        self.approximate_even_binf_double = _ApproximateEvenBinfDouble(self)

    def __del__(self):
        cdll = self.__class__.cdll
        if cdll is None:
            return
        cdll.delete_gm_to_dirac_short_double(self.gm_to_dirac_double)
        cdll.delete_gm_to_dirac_short_standard_normal_deviation_double(
            self.gm_to_dirac_snd_double
        )
        cdll.delete_gm_to_dirac_even_closed_form_double(self.gm_to_dirac_even_double)
        cdll.delete_gm_to_dirac_even_binf_double(self.gm_to_dirac_even_binf_double)
