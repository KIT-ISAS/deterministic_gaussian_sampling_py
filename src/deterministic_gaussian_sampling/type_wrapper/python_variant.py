# python_variant.py
from __future__ import annotations
from dataclasses import dataclass
import ctypes
import numpy as np

@dataclass
class ApproximationResultPy:
    success: bool
    result: "GslMinimizerResultPy"
    x: np.ndarray

    @staticmethod
    def from_ctypes(
        success: ctypes.c_bool,
        minimizer_result,
        x_ptr: ctypes.POINTER[ctypes.c_double],
        L: int,
        N: int,
    ) -> "ApproximationResultPy":
        from deterministic_gaussian_sampling.type_wrapper import ctypes_wrapper

        return ApproximationResultPy(
            bool(success),
            minimizer_result.to_py_type(),
            np.ctypeslib.as_array(x_ptr, shape=(L, N)),
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
    initalStepSize: float = 0.0
    stepTolerance: float = 0.0
    lastXtolAbs: float = 0.0
    lastXtolRel: float = 0.0
    lastFtolAbs: float = 0.0
    lastFtolRel: float = 0.0
    lastGtol: float = 0.0
    xtolAbs: float = 0.0
    xtolRel: float = 0.0
    ftolAbs: float = 0.0
    ftolRel: float = 0.0
    gtol: float = 0.0
    iterations: int = 0
    maxIterations: int = 0
    elapsedTimeMicro: int = 0

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
