# ctypes_wrapper.py
import ctypes
from typing import TYPE_CHECKING

# avoid importing python_variant at top-level to break circular imports
# (only import it inside functions when needed)

class GslMinimizerResultCTypes(ctypes.Structure):
    _fields_ = [
        ("initalStepSize", ctypes.c_double),
        ("stepTolerance", ctypes.c_double),
        ("lastXtolAbs", ctypes.c_double),
        ("lastXtolRel", ctypes.c_double),
        ("lastFtolAbs", ctypes.c_double),
        ("lastFtolRel", ctypes.c_double),
        ("lastGtol", ctypes.c_double),
        ("xtolAbs", ctypes.c_double),
        ("xtolRel", ctypes.c_double),
        ("ftolAbs", ctypes.c_double),
        ("ftolRel", ctypes.c_double),
        ("gtol", ctypes.c_double),
        ("iterations", ctypes.c_size_t),
        ("maxIterations", ctypes.c_size_t),
        ("elapsedTimeMicro", ctypes.c_size_t),
    ]

    @staticmethod
    def from_py_type(pyT: "deterministic_gaussian_sampling.type_wrapper.python_variant.GslMinimizerResultPy") -> "GslMinimizerResultCTypes":
        import deterministic_gaussian_sampling.type_wrapper.python_variant as python_variant
        return GslMinimizerResultCTypes(
            ctypes.c_double(pyT.initalStepSize),
            ctypes.c_double(pyT.stepTolerance),
            ctypes.c_double(pyT.lastXtolAbs),
            ctypes.c_double(pyT.lastXtolRel),
            ctypes.c_double(pyT.lastFtolAbs),
            ctypes.c_double(pyT.lastFtolRel),
            ctypes.c_double(pyT.lastGtol),
            ctypes.c_double(pyT.xtolAbs),
            ctypes.c_double(pyT.xtolRel),
            ctypes.c_double(pyT.ftolAbs),
            ctypes.c_double(pyT.ftolRel),
            ctypes.c_double(pyT.gtol),
            ctypes.c_size_t(pyT.iterations),
            ctypes.c_size_t(pyT.maxIterations),
            ctypes.c_size_t(pyT.elapsedTimeMicro),
        )

    def to_py_type(self):
        import deterministic_gaussian_sampling.type_wrapper.python_variant as python_variant
        return python_variant.GslMinimizerResultPy(
            float(self.initalStepSize),
            float(self.stepTolerance),
            float(self.lastXtolAbs),
            float(self.lastXtolRel),
            float(self.lastFtolAbs),
            float(self.lastFtolRel),
            float(self.lastGtol),
            float(self.xtolAbs),
            float(self.xtolRel),
            float(self.ftolAbs),
            float(self.ftolRel),
            float(self.gtol),
            int(self.iterations),
            int(self.maxIterations),
            int(self.elapsedTimeMicro),
        )


class ApproximateOptionsCTypes(ctypes.Structure):
    _fields_ = [
        ("xtolAbs", ctypes.c_double),
        ("xtolRel", ctypes.c_double),
        ("ftolAbs", ctypes.c_double),
        ("ftolRel", ctypes.c_double),
        ("gtol", ctypes.c_double),
        ("initialX", ctypes.c_bool),
        ("maxIterations", ctypes.c_size_t),
        ("verbose", ctypes.c_bool),
        ("bMax", ctypes.c_size_t),
    ]

    @staticmethod
    def from_py_type(pyT: "deterministic_gaussian_sampling.type_wrapper.python_variant.ApproximateOptionsPy") -> "ApproximateOptionsCTypes":
        import deterministic_gaussian_sampling.type_wrapper.python_variant as python_variant
        return ApproximateOptionsCTypes(
            ctypes.c_double(pyT.xtolAbs),
            ctypes.c_double(pyT.xtolRel),
            ctypes.c_double(pyT.ftolAbs),
            ctypes.c_double(pyT.ftolRel),
            ctypes.c_double(pyT.gtol),
            ctypes.c_bool(pyT.initialX),
            ctypes.c_size_t(pyT.maxIterations),
            ctypes.c_bool(pyT.verbose),
            ctypes.c_size_t(pyT.bMax),
        )

    def to_py_type(self):
        import deterministic_gaussian_sampling.type_wrapper.python_variant as python_variant
        return python_variant.ApproximateOptionsPy(
            float(self.xtolAbs),
            float(self.xtolRel),
            float(self.ftolAbs),
            float(self.ftolRel),
            float(self.gtol),
            bool(self.initialX),
            int(self.maxIterations),
            bool(self.verbose),
            int(self.bMax),
        )


wXCallbackCTypes = ctypes.CFUNCTYPE(
    None,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t,
    ctypes.c_size_t,
)
wXDCallbackCTypes = wXCallbackCTypes
