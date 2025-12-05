from pathlib import Path
import ctypes

import deterministic_gaussian_sampling.type_wrapper.ctypes_wrapper as ctypes_wrapper

def _setup_ctypes_dll(cdll: ctypes.CDLL) -> ctypes.CDLL:
    # dirac_to_dirac_approx_short_double
    cdll.create_dirac_to_dirac_approx_short_double.restype = ctypes.c_void_p
    cdll.create_dirac_to_dirac_approx_short_double.argtypes = []

    cdll.delete_dirac_to_dirac_approx_short_double.restype = None
    cdll.delete_dirac_to_dirac_approx_short_double.argtypes = [ctypes.c_void_p]

    cdll.dirac_to_dirac_approx_short_double_approximate.restype = ctypes.c_bool
    cdll.dirac_to_dirac_approx_short_double_approximate.argtypes = [
        ctypes.c_void_p,                            # instance
        ctypes.POINTER(ctypes.c_double),            # y
        ctypes.c_size_t,                            # M
        ctypes.c_size_t,                            # L
        ctypes.c_size_t,                            # N
        ctypes.c_size_t,                            # bMax
        ctypes.POINTER(ctypes.c_double),            # x
        ctypes.POINTER(ctypes.c_double),            # wX
        ctypes.POINTER(ctypes.c_double),            # wY
        ctypes.POINTER(ctypes_wrapper.GslMinimizerResultCTypes),   # result
        ctypes.POINTER(ctypes_wrapper.ApproximateOptionsCTypes)    # options
    ]

    # dirac_to_dirac_approx_short_function_double
    cdll.create_dirac_to_dirac_approx_short_function_double.restype = ctypes.c_void_p
    cdll.create_dirac_to_dirac_approx_short_function_double.argtypes = []

    cdll.delete_dirac_to_dirac_approx_short_function_double.restype = None
    cdll.delete_dirac_to_dirac_approx_short_function_double.argtypes = [ctypes.c_void_p]

    cdll.dirac_to_dirac_approx_short_function_double_approximate.restype = ctypes.c_bool
    cdll.dirac_to_dirac_approx_short_function_double_approximate.argtypes = [
        ctypes.c_void_p,                            # instance
        ctypes.POINTER(ctypes.c_double),            # y
        ctypes.c_size_t,                            # M
        ctypes.c_size_t,                            # L
        ctypes.c_size_t,                            # N
        ctypes.c_size_t,                            # bMax
        ctypes.POINTER(ctypes.c_double),            # x
        ctypes_wrapper.wXCallbackCTypes,                           # wX callback
        ctypes_wrapper.wXDCallbackCTypes,                          # wXD callback
        ctypes.POINTER(ctypes_wrapper.GslMinimizerResultCTypes),   # result
        ctypes.POINTER(ctypes_wrapper.ApproximateOptionsCTypes)    # options
    ]

    # dirac_to_dirac_approx_short_thread_double
    cdll.create_dirac_to_dirac_approx_short_thread_double.restype = ctypes.c_void_p
    cdll.create_dirac_to_dirac_approx_short_thread_double.argtypes = []

    cdll.delete_dirac_to_dirac_approx_short_thread_double.restype = None
    cdll.delete_dirac_to_dirac_approx_short_thread_double.argtypes = [ctypes.c_void_p]

    cdll.dirac_to_dirac_approx_short_thread_double_approximate.restype = ctypes.c_bool
    cdll.dirac_to_dirac_approx_short_thread_double_approximate.argtypes = [
        ctypes.c_void_p,                            # instance
        ctypes.POINTER(ctypes.c_double),            # y
        ctypes.c_size_t,                            # M
        ctypes.c_size_t,                            # L
        ctypes.c_size_t,                            # N
        ctypes.c_size_t,                            # bMax
        ctypes.POINTER(ctypes.c_double),            # x
        ctypes.POINTER(ctypes.c_double),            # wX
        ctypes.POINTER(ctypes.c_double),            # wY
        ctypes.POINTER(ctypes_wrapper.GslMinimizerResultCTypes),   # result
        ctypes.POINTER(ctypes_wrapper.ApproximateOptionsCTypes)    # options
    ]

    # gm_to_dirac_short_double
    cdll.create_gm_to_dirac_short_double.restype = ctypes.c_void_p
    cdll.create_gm_to_dirac_short_double.argtypes = []

    cdll.delete_gm_to_dirac_short_double.restype = None
    cdll.delete_gm_to_dirac_short_double.argtypes = [ctypes.c_void_p]

    cdll.gm_to_dirac_short_double_approximate.restype = ctypes.c_bool
    cdll.gm_to_dirac_short_double_approximate.argtypes = [
        ctypes.c_void_p,                            # instance
        ctypes.POINTER(ctypes.c_double),            # covDiag
        ctypes.c_size_t,                            # L
        ctypes.c_size_t,                            # N
        ctypes.c_size_t,                            # bMax
        ctypes.POINTER(ctypes.c_double),            # x
        ctypes.POINTER(ctypes.c_double),            # wX
        ctypes.POINTER(ctypes_wrapper.GslMinimizerResultCTypes),   # result
        ctypes.POINTER(ctypes_wrapper.ApproximateOptionsCTypes)    # options
    ]

    # gm_to_dirac_short_standard_normal_deviation_double
    cdll.create_gm_to_dirac_short_standard_normal_deviation_double.restype = ctypes.c_void_p
    cdll.create_gm_to_dirac_short_standard_normal_deviation_double.argtypes = []

    cdll.delete_gm_to_dirac_short_standard_normal_deviation_double.restype = None
    cdll.delete_gm_to_dirac_short_standard_normal_deviation_double.argtypes = [ctypes.c_void_p]

    cdll.gm_to_dirac_short_standard_normal_deviation_double_approximate.restype = ctypes.c_bool
    cdll.gm_to_dirac_short_standard_normal_deviation_double_approximate.argtypes = [
        ctypes.c_void_p,                            # instance
        ctypes.c_size_t,                            # L
        ctypes.c_size_t,                            # N
        ctypes.c_size_t,                            # bMax
        ctypes.POINTER(ctypes.c_double),            # x
        ctypes.POINTER(ctypes.c_double),            # wX
        ctypes.POINTER(ctypes_wrapper.GslMinimizerResultCTypes),   # result
        ctypes.POINTER(ctypes_wrapper.ApproximateOptionsCTypes)    # options
    ]

    return cdll

def load_dll() -> ctypes.CDLL:
    package_root = Path(__file__).resolve().parent.parent
    dll_path = package_root / "lib" / "windows" / "bin" / "libapproxLCD.dll"

    return _setup_ctypes_dll(ctypes.CDLL(dll_path))