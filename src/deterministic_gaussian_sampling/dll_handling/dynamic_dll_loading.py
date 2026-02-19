import ctypes
import platform
from pathlib import Path

import deterministic_gaussian_sampling.type_wrapper.ctypes_wrapper as ctypes_wrapper


def _setup_ctypes_dll(cdll: ctypes.CDLL) -> ctypes.CDLL:

    c_void_p = ctypes.c_void_p
    c_double_p = ctypes.POINTER(ctypes.c_double)
    c_size = ctypes.c_size_t
    c_bool = ctypes.c_bool

    gsl_result_p = ctypes.POINTER(ctypes_wrapper.GslMinimizerResultCTypes)
    approx_opts_p = ctypes.POINTER(ctypes_wrapper.ApproximateOptionsCTypes)

    # ============================================================ #
    # ============= DIRAC to DIRAC (standard double) ============= #
    # ============================================================ #

    cdll.create_dirac_to_dirac_approx_short_double.restype = c_void_p
    cdll.create_dirac_to_dirac_approx_short_double.argtypes = []

    cdll.delete_dirac_to_dirac_approx_short_double.restype = None
    cdll.delete_dirac_to_dirac_approx_short_double.argtypes = [c_void_p]

    cdll.dirac_to_dirac_approx_short_double_approximate.restype = c_bool
    cdll.dirac_to_dirac_approx_short_double_approximate.argtypes = [
        c_void_p,
        c_double_p,  # y
        c_size,      # M
        c_size,      # L
        c_size,      # N
        c_size,      # bMax
        c_double_p,  # x
        c_double_p,  # wX
        c_double_p,  # wY
        gsl_result_p,
        approx_opts_p,
    ]

    cdll.dirac_to_dirac_approx_short_double_modified_van_mises_distance_sq.restype = None
    cdll.dirac_to_dirac_approx_short_double_modified_van_mises_distance_sq.argtypes = [
        c_void_p,
        ctypes.POINTER(ctypes.c_double),  # result
        c_double_p,
        c_size,
        c_size,
        c_size,
        c_size,
        c_double_p,
        c_double_p,
        c_double_p,
    ]

    cdll.dirac_to_dirac_approx_short_double_modified_van_mises_distance_sq_derivative.restype = None
    cdll.dirac_to_dirac_approx_short_double_modified_van_mises_distance_sq_derivative.argtypes = [
        c_void_p,
        c_double_p,  # gradient
        c_double_p,
        c_size,
        c_size,
        c_size,
        c_size,
        c_double_p,
        c_double_p,
        c_double_p,
    ]

    # ============================================================ #
    # ============ DIRAC to DIRAC (function callback) ============ #
    # ============================================================ #

    cdll.create_dirac_to_dirac_approx_short_function_double.restype = c_void_p
    cdll.create_dirac_to_dirac_approx_short_function_double.argtypes = []

    cdll.delete_dirac_to_dirac_approx_short_function_double.restype = None
    cdll.delete_dirac_to_dirac_approx_short_function_double.argtypes = [c_void_p]

    cdll.dirac_to_dirac_approx_short_function_double_approximate.restype = c_bool
    cdll.dirac_to_dirac_approx_short_function_double_approximate.argtypes = [
        c_void_p,
        c_double_p,
        c_size,
        c_size,
        c_size,
        c_size,
        c_double_p,
        ctypes_wrapper.wXCallbackCTypes,
        ctypes_wrapper.wXDCallbackCTypes,
        gsl_result_p,
        approx_opts_p,
    ]

    cdll.dirac_to_dirac_approx_short_function_double_modified_van_mises_distance_sq.restype = None
    cdll.dirac_to_dirac_approx_short_function_double_modified_van_mises_distance_sq.argtypes = [
        c_void_p,
        ctypes.POINTER(ctypes.c_double),
        c_double_p,
        c_size,
        c_size,
        c_size,
        c_size,
        c_double_p,
        ctypes_wrapper.wXCallbackCTypes,
        ctypes_wrapper.wXDCallbackCTypes,
    ]

    cdll.dirac_to_dirac_approx_short_function_double_modified_van_mises_distance_sq_derivative.restype = None
    cdll.dirac_to_dirac_approx_short_function_double_modified_van_mises_distance_sq_derivative.argtypes = [
        c_void_p,
        c_double_p,
        c_double_p,
        c_size,
        c_size,
        c_size,
        c_size,
        c_double_p,
        ctypes_wrapper.wXCallbackCTypes,
        ctypes_wrapper.wXDCallbackCTypes,
    ]

    # ============================================================ #
    # ================= DIRAC to DIRAC (threaded) ================ #
    # ============================================================ #

    cdll.create_dirac_to_dirac_approx_short_thread_double.restype = c_void_p
    cdll.create_dirac_to_dirac_approx_short_thread_double.argtypes = []

    cdll.delete_dirac_to_dirac_approx_short_thread_double.restype = None
    cdll.delete_dirac_to_dirac_approx_short_thread_double.argtypes = [c_void_p]

    cdll.dirac_to_dirac_approx_short_thread_double_approximate.restype = c_bool
    cdll.dirac_to_dirac_approx_short_thread_double_approximate.argtypes = [
        c_void_p,
        c_double_p,
        c_size,
        c_size,
        c_size,
        c_size,
        c_double_p,
        c_double_p,
        c_double_p,
        gsl_result_p,
        approx_opts_p,
    ]

    cdll.dirac_to_dirac_approx_short_thread_double_modified_van_mises_distance_sq.restype = None
    cdll.dirac_to_dirac_approx_short_thread_double_modified_van_mises_distance_sq.argtypes = [
        c_void_p,
        ctypes.POINTER(ctypes.c_double),
        c_double_p,
        c_size,
        c_size,
        c_size,
        c_size,
        c_double_p,
        c_double_p,
        c_double_p,
    ]

    cdll.dirac_to_dirac_approx_short_thread_double_modified_van_mises_distance_sq_derivative.restype = None
    cdll.dirac_to_dirac_approx_short_thread_double_modified_van_mises_distance_sq_derivative.argtypes = [
        c_void_p,
        c_double_p,
        c_double_p,
        c_size,
        c_size,
        c_size,
        c_size,
        c_double_p,
        c_double_p,
        c_double_p,
    ]

    # ============================================================ #
    # ============ GAUSSIAN to DIRAC (full covariance) =========== #
    # ============================================================ #

    cdll.create_gm_to_dirac_short_double.restype = c_void_p
    cdll.create_gm_to_dirac_short_double.argtypes = []

    cdll.delete_gm_to_dirac_short_double.restype = None
    cdll.delete_gm_to_dirac_short_double.argtypes = [c_void_p]

    cdll.gm_to_dirac_short_double_approximate.restype = c_bool
    cdll.gm_to_dirac_short_double_approximate.argtypes = [
        c_void_p,
        c_double_p,  # sqrt eigenvalues
        c_size,
        c_size,
        c_size,
        c_double_p,
        c_double_p,
        gsl_result_p,
        approx_opts_p,
    ]

    cdll.gm_to_dirac_short_double_modified_van_mises_distance_sq.restype = None
    cdll.gm_to_dirac_short_double_modified_van_mises_distance_sq.argtypes = [
        c_void_p,
        c_double_p,
        ctypes.POINTER(ctypes.c_double),
        c_size,
        c_size,
        c_size,
        c_double_p,
        c_double_p,
    ]

    cdll.gm_to_dirac_short_double_modified_van_mises_distance_sq_derivative.restype = None
    cdll.gm_to_dirac_short_double_modified_van_mises_distance_sq_derivative.argtypes = [
        c_void_p,
        c_double_p,
        c_double_p,
        c_size,
        c_size,
        c_size,
        c_double_p,
        c_double_p,
    ]

    # ============================================================ #
    # ======= GAUSSIAN to DIRAC (standard normal deviation) ====== #
    # ============================================================ #

    cdll.create_gm_to_dirac_short_standard_normal_deviation_double.restype = c_void_p
    cdll.create_gm_to_dirac_short_standard_normal_deviation_double.argtypes = []

    cdll.delete_gm_to_dirac_short_standard_normal_deviation_double.restype = None
    cdll.delete_gm_to_dirac_short_standard_normal_deviation_double.argtypes = [c_void_p]

    cdll.gm_to_dirac_short_standard_normal_deviation_double_approximate.restype = c_bool
    cdll.gm_to_dirac_short_standard_normal_deviation_double_approximate.argtypes = [
        c_void_p,
        c_size,
        c_size,
        c_size,
        c_double_p,
        c_double_p,
        gsl_result_p,
        approx_opts_p,
    ]

    cdll.gm_to_dirac_short_standard_normal_deviation_double_modified_van_mises_distance_sq.restype = None
    cdll.gm_to_dirac_short_standard_normal_deviation_double_modified_van_mises_distance_sq.argtypes = [
        c_void_p,
        ctypes.POINTER(ctypes.c_double),
        c_size,
        c_size,
        c_size,
        c_double_p,
        c_double_p,
    ]

    cdll.gm_to_dirac_short_standard_normal_deviation_double_modified_van_mises_distance_sq_derivative.restype = None
    cdll.gm_to_dirac_short_standard_normal_deviation_double_modified_van_mises_distance_sq_derivative.argtypes = [
        c_void_p,
        c_double_p,
        c_size,
        c_size,
        c_size,
        c_double_p,
        c_double_p,
    ]

    return cdll


def load_dll() -> ctypes.CDLL:
    package_root = Path(__file__).resolve().parent.parent

    system = platform.system()
    if system == "Windows":
        dll_rel = Path("lib") / "windows" / "bin" / "libapproxLCD.dll"
    elif system == "Linux":
        dll_rel = Path("lib") / "linux" / "bin" / "libapproxLCD.so"
    elif system == "Darwin":
        raise RuntimeError(f"MacOS is currently not supported ):")
        dll_rel = Path("lib") / "macos" / "bin" / "libapproxLCD.dylib"
    else:
        raise RuntimeError(f"Unsupported OS: {system}")
    
    dll_path = package_root / dll_rel

    return _setup_ctypes_dll(ctypes.CDLL(str(dll_path), use_errno=True))
