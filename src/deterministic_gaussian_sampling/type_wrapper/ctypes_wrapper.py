import ctypes
import deterministic_gaussian_sampling.type_wrapper.python_variant as python_variant

class GslMinimizerResultCTypes(ctypes.Structure):
    _fields_ = [
        ("initalStepSize", ctypes.c_double),    # initialStepSize
        ("stepTolerance", ctypes.c_double),     # stepTolerance
        ("lastXtolAbs", ctypes.c_double),       # lastXtolAbs
        ("lastXtolRel", ctypes.c_double),       # lastFtolRel
        ("lastFtolAbs", ctypes.c_double),       # lastFtolAbs
        ("lastFtolRel", ctypes.c_double),       # lastFtolRel
        ("lastGtol", ctypes.c_double),          # lastGtol
        ("xtolAbs", ctypes.c_double),           # xtolAbs
        ("xtolRel", ctypes.c_double),           # xtolRel
        ("ftolAbs", ctypes.c_double),           # ftolAbs
        ("ftolRel", ctypes.c_double),           # ftolRel
        ("gtol", ctypes.c_double),              # gtol
        ("iterations", ctypes.c_size_t),        # iterations
        ("maxIterations", ctypes.c_size_t),     # maxIterations
        ("elapsedTimeMicro", ctypes.c_size_t),  # elapsedTimeMicro
    ]

    @staticmethod
    def from_py_type(pyT: python_variant.GslMinimizerResultPy) -> GslMinimizerResultCTypes:
        return GslMinimizerResultCTypes(
            ctypes.c_double(pyT.initalStepSize),    # initialStepSize
            ctypes.c_double(pyT.stepTolerance),     # stepTolerance
            ctypes.c_double(pyT.lastXtolAbs),       # lastXtolAbs
            ctypes.c_double(pyT.lastXtolRel),       # lastFtolRel
            ctypes.c_double(pyT.lastFtolAbs),       # lastFtolAbs
            ctypes.c_double(pyT.lastFtolRel),       # lastFtolRel
            ctypes.c_double(pyT.lastGtol),          # lastGtol
            ctypes.c_double(pyT.xtolAbs),           # xtolAbs
            ctypes.c_double(pyT.xtolRel),           # xtolRel
            ctypes.c_double(pyT.ftolAbs),           # ftolAbs
            ctypes.c_double(pyT.ftolRel),           # ftolRel
            ctypes.c_double(pyT.gtol),              # gtol
            ctypes.c_size_t(pyT.iterations),        # iterations
            ctypes.c_size_t(pyT.maxIterations),     # maxIterations
            ctypes.c_size_t(pyT.elapsedTimeMicro)   # elapsedTimeMicro
        )
    
    def to_py_type(self) -> python_variant.GslMinimizerResultPy:
        return python_variant.GslMinimizerResultPy(
            float(self.initalStepSize), # initialStepSize
            float(self.stepTolerance),  # stepTolerance
            float(self.lastXtolAbs),    # lastXtolAbs
            float(self.lastXtolRel),    # lastFtolRel
            float(self.lastFtolAbs),    # lastFtolAbs
            float(self.lastFtolRel),    # lastFtolRel
            float(self.lastGtol),       # lastGtol
            float(self.xtolAbs),        # xtolAbs
            float(self.xtolRel),        # xtolRel
            float(self.ftolAbs),        # ftolAbs
            float(self.ftolRel),        # ftolRel
            float(self.gtol),           # gtol
            int(self.iterations),       # iterations
            int(self.maxIterations),    # maxIterations
            int(self.elapsedTimeMicro)  # elapsedTimeMicro
        )

class ApproximateOptionsCTypes(ctypes.Structure):
    _fields_ = [
        ("xtolAbs", ctypes.c_double),       # xtolAbs
        ("xtolRel", ctypes.c_double),       # xtolRel
        ("ftolAbs", ctypes.c_double),       # ftolAbs
        ("ftolRel", ctypes.c_double),       # ftolRel
        ("gtol", ctypes.c_double),          # gtol
        ("initialX", ctypes.c_bool),        # intialX
        ("maxIterations", ctypes.c_size_t), # maxIterations
        ("verbose", ctypes.c_bool),         # verbose
    ]

    @staticmethod
    def from_py_type(pyT: python_variant.ApproximateOptionsPy) -> ApproximateOptionsCTypes:
        return ApproximateOptionsCTypes(
            ctypes.c_double(pyT.xtolAbs),       # xtolAbs
            ctypes.c_double(pyT.xtolRel),       # xtolRel
            ctypes.c_double(pyT.ftolAbs),       # ftolAbs
            ctypes.c_double(pyT.ftolRel),       # ftolRel
            ctypes.c_double(pyT.gtol),          # gtol
            ctypes.c_bool(pyT.initialX),        # intialX
            ctypes.c_size_t(pyT.maxIterations), # maxIteration
            ctypes.c_bool(pyT.verbose)          # verbose
        )
    
    def to_py_type(self) -> python_variant.ApproximateOptionsPy:
        return python_variant.ApproximateOptionsPy(
            float(self.xtolAbs),        # xtolAbs
            float(self.xtolRel),        # xtolRel
            float(self.ftolAbs),        # ftolAbs
            float(self.ftolRel),        # ftolRel
            float(self.gtol),           # gtol
            bool(self.initialX),        # intialX
            int(self.maxIterations),    # maxIteration
            bool(self.verbose)          # verbose
        )

wXCallbackCTypes = ctypes.CFUNCTYPE(
    None,                               # return
    ctypes.POINTER(ctypes.c_double),    # x
    ctypes.POINTER(ctypes.c_double),    # res
    ctypes.c_size_t,                    # L
    ctypes.c_size_t                     # N
)
wXDCallbackCTypes = wXCallbackCTypes