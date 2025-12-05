from dataclasses import dataclass
import ctypes
import numpy
import deterministic_gaussian_sampling.type_wrapper.python_variant as python_variant
import deterministic_gaussian_sampling.type_wrapper.ctypes_wrapper as ctypes_wrapper


@dataclass
class ApproximationResultPy:
    success: bool
    result: python_variant.GslMinimizerResultPy
    x: numpy.ndarray

    @staticmethod
    def from_ctypes(
        success: ctypes.c_bool,
        minimizer_result: ctypes_wrapper.GslMinimizerResultCTypes,
        x_ptr: ctypes.POINTER[ctypes.c_double],
        L: int,
        N: int,
    ) -> "ApproximationResultPy":
        return ApproximationResultPy(
            bool(success),
            minimizer_result.to_py_type(),
            numpy.ctypeslib.as_array(x_ptr, shape=(L, N)),
        )

    def __str__(self):
        lines = []
        lines.append(f"Success: {self.success}")
        lines.append(str(self.result))
        lines.append(f"x: {str(self.x)}")
        return "\n".join(lines)

    __repr__ = __str__


@dataclass
class GslMinimizerResultPy:
    initalStepSize: float
    stepTolerance: float
    lastXtolAbs: float
    lastXtolRel: float
    lastFtolAbs: float
    lastFtolRel: float
    lastGtol: float
    xtolAbs: float
    xtolRel: float
    ftolAbs: float
    ftolRel: float
    gtol: float
    iterations: int
    maxIterations: int
    elapsedTimeMicro: int

    def __init__(
        self,
        initalStepSize=0.0,
        stepTolerance=0.0,
        lastXtolAbs=0.0,
        lastXtolRel=0.0,
        lastFtolAbs=0.0,
        lastFtolRel=0.0,
        lastGtol=0.0,
        xtolAbs=0.0,
        xtolRel=0.0,
        ftolAbs=0.0,
        ftolRel=0.0,
        gtol=0.0,
        iterations=0,
        maxIterations=0,
        elapsedTimeMicro=0,
    ):
        self.initalStepSize = initalStepSize
        self.stepTolerance = stepTolerance
        self.lastXtolAbs = lastXtolAbs
        self.lastXtolRel = lastXtolRel
        self.lastFtolAbs = lastFtolAbs
        self.lastFtolRel = lastFtolRel
        self.lastGtol = lastGtol
        self.xtolAbs = xtolAbs
        self.xtolRel = xtolRel
        self.ftolAbs = ftolAbs
        self.ftolRel = ftolRel
        self.gtol = gtol
        self.iterations = iterations
        self.maxIterations = maxIterations
        self.elapsedTimeMicro = elapsedTimeMicro

    @staticmethod
    def _format_elapsed_time(elapsedMicro):
        if elapsedMicro < 1_000:
            return f"{elapsedMicro:.1f} µs"
        if elapsedMicro < 1_000_000:
            return f"{(elapsedMicro / 1_000):.3f} ms"
        if elapsedMicro < 60_000_000:
            return f"{(elapsedMicro / 1_000_000):.3f} s"
        return f"{(elapsedMicro / 60_000_000):.3f} m, {((elapsedMicro % 60_000_000) / 1_000_000):.2f} s"

    @staticmethod
    def _print_comp_symbol(a, b):
        if a < b:
            return "<"
        if a > b:
            return ">"
        return "="

    def __str__(self):
        lines = []
        lines.append(f"GslMinimizerResultPy:")
        lines.append(f"   initialStepSize: {self.initalStepSize}")
        lines.append(f"   stepTolerance:   {self.stepTolerance}")
        lines.append(
            f"   |x - x'|:               {self.lastXtolAbs} {self._print_comp_symbol(self.lastXtolAbs, self.xtolAbs)} {self.xtolAbs}"
        )
        lines.append(
            f"   |x - x'|/|x'|:          {self.lastXtolRel} {self._print_comp_symbol(self.lastXtolRel, self.xtolRel)} {self.xtolRel}"
        )
        lines.append(
            f"   |f(x) - f(x')|:         {self.lastFtolAbs} {self._print_comp_symbol(self.lastFtolAbs, self.ftolAbs)} {self.ftolAbs}"
        )
        lines.append(
            f"   |f(x) - f(x')|/|f(x')|: {self.lastFtolRel} {self._print_comp_symbol(self.lastFtolRel, self.ftolRel)} {self.ftolRel}"
        )
        lines.append(
            f"   |g(x)|:                 {self.lastGtol} {self._print_comp_symbol(self.lastGtol, self.gtol)} {self.gtol}"
        )
        lines.append(f"   iterations: {self.iterations} of {self.maxIterations}")
        lines.append(
            f"   timeTaken: {self._format_elapsed_time(self.elapsedTimeMicro)}"
        )
        return "\n".join(lines)

    __repr__ = __str__


@dataclass
class ApproximateOptionsPy:
    xtolAbs: float
    xtolRel: float
    ftolAbs: float
    ftolRel: float
    gtol: float
    initialX: bool
    maxIterations: int
    verbose: bool

    def __str__(self):
        lines = []
        lines.append(f"ApproximateOptionsPy:")
        lines.append(f"   xtolAbs: {self.xtolAbs}")
        lines.append(f"   xtolRel:   {self.xtolRel}")
        lines.append(f"   ftolAbs:   {self.ftolAbs}")
        lines.append(f"   ftolRel:   {self.ftolRel}")
        lines.append(f"   gtol:   {self.gtol}")
        lines.append(f"   initialX:   {self.initialX}")
        lines.append(f"   maxIterations:   {self.maxIterations}")
        lines.append(f"   verbose:   {self.verbose}")
        return "\n".join(lines)

    __repr__ = __str__
