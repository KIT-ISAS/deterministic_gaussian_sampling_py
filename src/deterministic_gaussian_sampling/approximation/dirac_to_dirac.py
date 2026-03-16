import ctypes
import numpy
from typing import Optional
import deterministic_gaussian_sampling.type_wrapper.python_variant as python_variant
import deterministic_gaussian_sampling.type_wrapper.ctypes_wrapper as ctypes_wrapper
from .base_approximation import BaseApproximation

class _ApproximateDouble:
    """
    Single-threaded Dirac-to-Dirac reduction (double precision).

    Callable interface for reducing an M-point Dirac mixture
    to L optimized Dirac points in N dimensions.
    """
    def __init__(self, parent):
        self._parent = parent

    def __call__(
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
        """
        Perform Dirac-to-Dirac reduction.

        Parameters
        ----------
        y : numpy.ndarray
            Input Dirac locations (shape: M x N).
        M : int
            Number of input components.
        L : int
            Number of reduced components.
        N : int
            Dimension.
        x : numpy.ndarray
            Initial guess (L x N). Overwritten with result.
        wX : numpy.ndarray, optional
            Weights of reduced mixture (size L).
        wY : numpy.ndarray, optional
            Weights of input mixture (size M).
        options : ApproximateOptionsPy, optional
            Optimization parameters.

        Returns
        -------
        ApproximationResultPy
            Result object containing success flag,
            minimizer information and optimized points.
        """

        cdll = self._parent.cdll
        if cdll is None:
            raise OSError("C++-Library was not loaded. Unable to continue!!!")
        yChecked = self._parent._check_numpy_ndarray(y, M, N)
        xChecked = self._parent._check_numpy_ndarray(x, L, N)
        wXChecked = self._parent._check_weights(wX, L)
        wYChecked = self._parent._check_weights(wY, M)
        minimizer_result = ctypes_wrapper.GslMinimizerResultCTypes()
        success: ctypes.c_bool = cdll.dirac_to_dirac_approx_short_double_approximate(
            self._parent.d2d_short_double,
            yChecked.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_size_t(M),
            ctypes.c_size_t(L),
            ctypes.c_size_t(N),
            xChecked.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            None if wXChecked is None else wXChecked.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            None if wYChecked is None else wYChecked.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
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

    def modified_van_mises_distance_sq(
        self,
        y: numpy.ndarray,
        M: int,
        L: int,
        N: int,
        x: numpy.ndarray,
        wX: Optional[numpy.ndarray] = None,
        wY: Optional[numpy.ndarray] = None
    ) -> float:
        """
        Compute squared modified van Mises distance.

        Parameters
        ----------
        y : numpy.ndarray
            Input Dirac locations (M x N).
        x : numpy.ndarray
            Reduced Dirac locations (L x N).

        Returns
        -------
        float
            Squared distance value.
        """
        cdll = self._parent.cdll
        if cdll is None:
            raise OSError("C++-Library was not loaded. Unable to continue!!!")
        yChecked = self._parent._check_numpy_ndarray(y, M, N)
        xChecked = self._parent._check_numpy_ndarray(x, L, N)
        wXChecked = self._parent._check_weights(wX, L)
        wYChecked = self._parent._check_weights(wY, M)
        result = ctypes.c_double(0.0)
        cdll.dirac_to_dirac_approx_short_double_modified_van_mises_distance_sq(
            self._parent.d2d_short_double,
            ctypes.byref(result),
            yChecked.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_size_t(M),
            ctypes.c_size_t(L),
            ctypes.c_size_t(N),
            xChecked.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            None if wXChecked is None else wXChecked.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            None if wYChecked is None else wYChecked.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )
        return result.value

    def modified_van_mises_distance_sq_derivative(
        self,
        y: numpy.ndarray,
        M: int,
        L: int,
        N: int,
        x: numpy.ndarray,
        wX: Optional[numpy.ndarray] = None,
        wY: Optional[numpy.ndarray] = None
    ) -> numpy.ndarray:
        """
        Compute analytical gradient of squared distance.

        Parameters
        ----------
        y : numpy.ndarray
            Input Dirac locations (M x N).
        x : numpy.ndarray
            Reduced Dirac locations (L x N).

        Returns
        -------
        numpy.ndarray
            Gradient array of shape (L x N).
        """
        cdll = self._parent.cdll
        if cdll is None:
            raise OSError("C++-Library was not loaded. Unable to continue!!!")
        yChecked = self._parent._check_numpy_ndarray(y, M, N)
        xChecked = self._parent._check_numpy_ndarray(x, L, N)
        wXChecked = self._parent._check_weights(wX, L)
        wYChecked = self._parent._check_weights(wY, M)
        gradient = numpy.zeros((L, N))
        gradientChecked = self._parent._check_numpy_ndarray(gradient, L, N)
        cdll.dirac_to_dirac_approx_short_double_modified_van_mises_distance_sq_derivative(
            self._parent.d2d_short_double,
            gradientChecked.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            yChecked.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_size_t(M),
            ctypes.c_size_t(L),
            ctypes.c_size_t(N),
            xChecked.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            None if wXChecked is None else wXChecked.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            None if wYChecked is None else wYChecked.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )
        return gradient

class _ApproximateFunctionDouble:
    """
    Dirac-to-Dirac reduction with user-defined weight functions.

    The reduced mixture weights are computed using
    Python callback functions.
    """
    def __init__(self, parent):
        self._parent = parent

    def __call__(
        self,
        y: numpy.ndarray,
        M: int,
        L: int,
        N: int,
        x: numpy.ndarray,
        wX: python_variant.wXCallbackPythonType,
        wXD: python_variant.wXDCallbackPythonType,
        options: Optional[python_variant.ApproximateOptionsPy] = None,
    ) -> python_variant.ApproximationResultPy:
        """
        Perform reduction using custom weight functions.

        Parameters
        ----------
        wX : callable
            Weight function callback.
        wXD : callable
            Derivative of weight function.

        Returns
        -------
        ApproximationResultPy
            Optimization result.
        """
        cdll = self._parent.cdll
        if cdll is None:
            raise OSError("C++-Library was not loaded. Unable to continue!!!")
        yChecked = self._parent._check_numpy_ndarray(y, M, N)
        xChecked = self._parent._check_numpy_ndarray(x, L, N)
        minimizer_result = ctypes_wrapper.GslMinimizerResultCTypes()
        wX = python_variant.wx_callback_python_wrapper(wX)
        wXD = python_variant.wxd_callback_python_wrapper(wXD)
        success = cdll.dirac_to_dirac_approx_short_function_double_approximate(
            self._parent.d2d_func_double,
            yChecked.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_size_t(M),
            ctypes.c_size_t(L),
            ctypes.c_size_t(N),
            xChecked.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            wX,
            wXD,
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
    
    def modified_van_mises_distance_sq(
        self,
        y: numpy.ndarray,
        M: int,
        L: int,
        N: int,
        x: numpy.ndarray,
        wX: python_variant.wXCallbackPythonType,
        wXD: python_variant.wXDCallbackPythonType
    ) -> float:
        """
        Compute squared modified van Mises distance.

        Parameters
        ----------
        y : numpy.ndarray
            Input Dirac locations (M x N).
        x : numpy.ndarray
            Reduced Dirac locations (L x N).

        Returns
        -------
        float
            Squared distance value.
        """
        cdll = self._parent.cdll
        if cdll is None:
            raise OSError("C++-Library was not loaded. Unable to continue!!!")
        yChecked = self._parent._check_numpy_ndarray(y, M, N)
        xChecked = self._parent._check_numpy_ndarray(x, L, N)
        result = ctypes.c_double(0.0)
        cdll.dirac_to_dirac_approx_short_function_double_modified_van_mises_distance_sq(
            self._parent.d2d_func_double,
            ctypes.byref(result),
            yChecked.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_size_t(M),
            ctypes.c_size_t(L),
            ctypes.c_size_t(N),
            xChecked.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            wX,
            wXD
        )
        return result.value
    
    def modified_van_mises_distance_sq_derivative(
        self,
        y: numpy.ndarray,
        M: int,
        L: int,
        N: int,
        x: numpy.ndarray,
        wX: python_variant.wXCallbackPythonType,
        wXD: python_variant.wXDCallbackPythonType
    ) -> numpy.ndarray:
        """
        Compute analytical gradient of squared distance.

        Parameters
        ----------
        y : numpy.ndarray
            Input Dirac locations (M x N).
        x : numpy.ndarray
            Reduced Dirac locations (L x N).

        Returns
        -------
        numpy.ndarray
            Gradient array of shape (L x N).
        """
        cdll = self._parent.cdll
        if cdll is None:
            raise OSError("C++-Library was not loaded. Unable to continue!!!")
        yChecked = self._parent._check_numpy_ndarray(y, M, N)
        xChecked = self._parent._check_numpy_ndarray(x, L, N)
        gradient = numpy.zeros((L, N))
        gradientChecked = self._parent._check_numpy_ndarray(gradient, L, N)
        cdll.dirac_to_dirac_approx_short_function_double_modified_van_mises_distance_sq_derivative(
            self._parent.d2d_func_double,
            gradientChecked.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            yChecked.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_size_t(M),
            ctypes.c_size_t(L),
            ctypes.c_size_t(N),
            xChecked.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            wX,
            wXD
        )
        return gradient

class _ApproximateThreadDouble:
    """
    Multi-threaded Dirac-to-Dirac reduction (double precision).

    Same functionality as the single-threaded version,
    but internally parallelized.
    """
    def __init__(self, parent):
        self._parent = parent

    def __call__(
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
        """
        Perform Dirac-to-Dirac reduction.

        Parameters
        ----------
        y : numpy.ndarray
            Input Dirac locations (shape: M x N).
        M : int
            Number of input components.
        L : int
            Number of reduced components.
        N : int
            Dimension.
        x : numpy.ndarray
            Initial guess (L x N). Overwritten with result.
        wX : numpy.ndarray, optional
            Weights of reduced mixture (size L).
        wY : numpy.ndarray, optional
            Weights of input mixture (size M).
        options : ApproximateOptionsPy, optional
            Optimization parameters.

        Returns
        -------
        ApproximationResultPy
            Result object containing success flag,
            minimizer information and optimized points.
        """
        cdll = self._parent.cdll
        if cdll is None:
            raise OSError("C++-Library was not loaded. Unable to continue!!!")
        yChecked = self._parent._check_numpy_ndarray(y, M, N)
        xChecked = self._parent._check_numpy_ndarray(x, L, N)
        wXChecked = self._parent._check_weights(wX, L)
        wYChecked = self._parent._check_weights(wY, M)
        minimizer_result = ctypes_wrapper.GslMinimizerResultCTypes()
        success = cdll.dirac_to_dirac_approx_short_thread_double_approximate(
            self._parent.d2d_thread_double,
            yChecked.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_size_t(M),
            ctypes.c_size_t(L),
            ctypes.c_size_t(N),
            xChecked.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            None if wXChecked is None else wXChecked.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            None if wYChecked is None else wYChecked.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
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

    def modified_van_mises_distance_sq(
        self,
        y: numpy.ndarray,
        M: int,
        L: int,
        N: int,
        x: numpy.ndarray,
        wX: Optional[numpy.ndarray] = None,
        wY: Optional[numpy.ndarray] = None
    ) -> float:
        """
        Compute squared modified van Mises distance.

        Parameters
        ----------
        y : numpy.ndarray
            Input Dirac locations (M x N).
        x : numpy.ndarray
            Reduced Dirac locations (L x N).

        Returns
        -------
        float
            Squared distance value.
        """
        cdll = self._parent.cdll
        if cdll is None:
            raise OSError("C++-Library was not loaded. Unable to continue!!!")
        yChecked = self._parent._check_numpy_ndarray(y, M, N)
        xChecked = self._parent._check_numpy_ndarray(x, L, N)
        wXChecked = self._parent._check_weights(wX, L)
        wYChecked = self._parent._check_weights(wY, M)
        result = ctypes.c_double(0.0)
        cdll.dirac_to_dirac_approx_short_thread_double_modified_van_mises_distance_sq(
            self._parent.d2d_thread_double,
            ctypes.byref(result),
            yChecked.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_size_t(M),
            ctypes.c_size_t(L),
            ctypes.c_size_t(N),
            xChecked.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            None if wXChecked is None else wXChecked.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            None if wYChecked is None else wYChecked.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )
        return result.value
    
    def modified_van_mises_distance_sq_derivative(
        self,
        y: numpy.ndarray,
        M: int,
        L: int,
        N: int,
        x: numpy.ndarray,
        wX: Optional[numpy.ndarray] = None,
        wY: Optional[numpy.ndarray] = None
    ) -> numpy.ndarray:
        """
        Compute analytical gradient of squared distance.

        Parameters
        ----------
        y : numpy.ndarray
            Input Dirac locations (M x N).
        x : numpy.ndarray
            Reduced Dirac locations (L x N).

        Returns
        -------
        numpy.ndarray
            Gradient array of shape (L x N).
        """
        cdll = self._parent.cdll
        if cdll is None:
            raise OSError("C++-Library was not loaded. Unable to continue!!!")
        yChecked = self._parent._check_numpy_ndarray(y, M, N)
        xChecked = self._parent._check_numpy_ndarray(x, L, N)
        wXChecked = self._parent._check_weights(wX, L)
        wYChecked = self._parent._check_weights(wY, M)
        gradient = numpy.zeros((L, N))
        gradientChecked = self._parent._check_numpy_ndarray(gradient, L, N)
        cdll.dirac_to_dirac_approx_short_thread_double_modified_van_mises_distance_sq_derivative(
            self._parent.d2d_thread_double,
            gradientChecked.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            yChecked.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_size_t(M),
            ctypes.c_size_t(L),
            ctypes.c_size_t(N),
            xChecked.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            None if wXChecked is None else wXChecked.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            None if wYChecked is None else wYChecked.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )
        return gradient

class DiracToDiracApproximation(BaseApproximation):
    """
    Dirac-to-Dirac reduction interface.

    Python wrapper around the C++ Dirac mixture reduction backend.

    This class provides:

    - Single-threaded reduction
    - Multi-threaded reduction
    - Reduction with user-defined weight functions
    - Distance evaluation
    - Analytical gradient computation

    All heavy computations are performed in the C++ core library.
    """
    def __init__(self):
        super().__init__()
        cdll = self.__class__.cdll
        if cdll is None:
            raise OSError("C++-Library was not loaded. Unable to continue!!!")
        self.d2d_short_double = cdll.create_dirac_to_dirac_approx_short_double()
        self.d2d_func_double = cdll.create_dirac_to_dirac_approx_short_function_double()
        self.d2d_thread_double = cdll.create_dirac_to_dirac_approx_short_thread_double()

        self.approximate_double = _ApproximateDouble(self)
        self.approximate_function_double = _ApproximateFunctionDouble(self)
        self.approximate_thread_double = _ApproximateThreadDouble(self)

    def __del__(self):
        cdll = self.__class__.cdll
        if cdll is None:
            return
        cdll.delete_dirac_to_dirac_approx_short_double(self.d2d_short_double)
        cdll.delete_dirac_to_dirac_approx_short_function_double(self.d2d_func_double)
        cdll.delete_dirac_to_dirac_approx_short_thread_double(self.d2d_thread_double)
    