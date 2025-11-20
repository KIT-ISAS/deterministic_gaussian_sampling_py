import os
import ctypes

from ctypes import Structure, c_double, c_size_t, c_bool, POINTER

class GslminimizerResult(Structure):
    _fields_ = [
        ("initalStepSize", c_double),
        ("stepTolerance", c_double),
        ("lastXtolAbs", c_double),
        ("lastXtolRel", c_double),
        ("lastFtolAbs", c_double),
        ("lastFtolRel", c_double),
        ("lastGtol", c_double),
        ("xtolAbs", c_double),
        ("xtolRel", c_double),
        ("ftolAbs", c_double),
        ("ftolRel", c_double),
        ("gtol", c_double),
        ("iterations", c_size_t),
        ("maxIterations", c_size_t),
        ("elapsedTimeMicro", c_size_t),
    ]

    def __str__(self):
        fields = [(name, getattr(self, name)) for name, _ in self._fields_]
        lines = [f"{name:20} = {value}" for name, value in fields]
        return "\n".join(lines)

    __repr__ = __str__

class ApproximateOptions(Structure):
    _fields_ = [
        ("xtolAbs", c_double),
        ("xtolRel", c_double),
        ("ftolAbs", c_double),
        ("ftolRel", c_double),
        ("gtol", c_double),
        ("initialX", c_bool),
        ("maxIterations", c_size_t),
        ("verbose", c_bool),
    ]

def _setup_ctypes_dll(cdll: ctypes.CDLL) -> ctypes.CDLL:
    # dirac_to_dirac_approx_short_double
    cdll.create_dirac_to_dirac_approx_short_double.restype = ctypes.c_void_p
    cdll.create_dirac_to_dirac_approx_short_double.argtypes = []

    cdll.delete_dirac_to_dirac_approx_short_double.restype = None
    cdll.delete_dirac_to_dirac_approx_short_double.argtypes = [ctypes.c_void_p]

    cdll.dirac_to_dirac_approx_short_double_approximate.restype = c_bool
    cdll.dirac_to_dirac_approx_short_double_approximate.argtypes = [
        ctypes.c_void_p,           # instance
        POINTER(c_double),         # y
        c_size_t, c_size_t, c_size_t,  # M, L, N
        c_size_t,                  # bMax
        POINTER(c_double),         # x
        POINTER(c_double),         # wX
        POINTER(c_double),         # wY
        POINTER(GslminimizerResult),  # result
        POINTER(ApproximateOptions)    # options
    ]

    # dirac_to_dirac_approx_short_float
    cdll.create_dirac_to_dirac_approx_short_float.restype = ctypes.c_void_p
    cdll.create_dirac_to_dirac_approx_short_float.argtypes = []

    cdll.delete_dirac_to_dirac_approx_short_float.restype = None
    cdll.delete_dirac_to_dirac_approx_short_float.argtypes = [ctypes.c_void_p]

    # dirac_to_dirac_approx_short_function_double
    cdll.create_dirac_to_dirac_approx_short_function_double.restype = ctypes.c_void_p
    cdll.create_dirac_to_dirac_approx_short_function_double.argtypes = []

    cdll.delete_dirac_to_dirac_approx_short_function_double.restype = None
    cdll.delete_dirac_to_dirac_approx_short_function_double.argtypes = [ctypes.c_void_p]

    # dirac_to_dirac_approx_short_thread_double
    cdll.create_dirac_to_dirac_approx_short_thread_double.restype = ctypes.c_void_p
    cdll.create_dirac_to_dirac_approx_short_thread_double.argtypes = []

    cdll.delete_dirac_to_dirac_approx_short_thread_double.restype = None
    cdll.delete_dirac_to_dirac_approx_short_thread_double.argtypes = [ctypes.c_void_p]

    # dirac_to_dirac_approx_short_thread_float
    cdll.create_dirac_to_dirac_approx_short_thread_float.restype = ctypes.c_void_p
    cdll.create_dirac_to_dirac_approx_short_thread_float.argtypes = []

    cdll.delete_dirac_to_dirac_approx_short_thread_float.restype = None
    cdll.delete_dirac_to_dirac_approx_short_thread_float.argtypes = [ctypes.c_void_p]

    # gm_to_dirac_short_double
    cdll.create_gm_to_dirac_short_double.restype = ctypes.c_void_p
    cdll.create_gm_to_dirac_short_double.argtypes = []

    cdll.delete_gm_to_dirac_short_double.restype = None
    cdll.delete_gm_to_dirac_short_double.argtypes = [ctypes.c_void_p]

    # gm_to_dirac_short_float
    cdll.create_gm_to_dirac_short_float.restype = ctypes.c_void_p
    cdll.create_gm_to_dirac_short_float.argtypes = []

    cdll.delete_gm_to_dirac_short_float.restype = None
    cdll.delete_gm_to_dirac_short_float.argtypes = [ctypes.c_void_p]

    # gm_to_dirac_short_standard_normal_deviation_double
    cdll.create_gm_to_dirac_short_standard_normal_deviation_double.restype = ctypes.c_void_p
    cdll.create_gm_to_dirac_short_standard_normal_deviation_double.argtypes = []

    cdll.delete_gm_to_dirac_short_standard_normal_deviation_double.restype = None
    cdll.delete_gm_to_dirac_short_standard_normal_deviation_double.argtypes = [ctypes.c_void_p]

    # gm_to_dirac_short_standard_normal_deviation_float
    cdll.create_gm_to_dirac_short_standard_normal_deviation_float.restype = ctypes.c_void_p
    cdll.create_gm_to_dirac_short_standard_normal_deviation_float.argtypes = []

    cdll.delete_gm_to_dirac_short_standard_normal_deviation_float.restype = None
    cdll.delete_gm_to_dirac_short_standard_normal_deviation_float.argtypes = [ctypes.c_void_p]
    
    return cdll

def load_dll() -> ctypes.CDLL:
    dll_dir = os.path.dirname(__file__)
    lib_path = os.path.join(dll_dir, "lib", "win64", "bin", "libapproxLCD.dll")
    return _setup_ctypes_dll(ctypes.CDLL(lib_path))