import ctypes

class GaussianToDiracApproximation:
    def __init__(self, cdll: ctypes.CDLL):
        self.cdll = cdll
        self.gm_to_dirac_double = cdll.create_gm_to_dirac_short_double()
        self.gm_to_dirac_float = cdll.create_gm_to_dirac_short_float()
        self.gm_to_dirac_snd_double = cdll.create_gm_to_dirac_short_standard_normal_deviation_double()
        self.gm_to_dirac_snd_float = cdll.create_gm_to_dirac_short_standard_normal_deviation_float()

    def __del__(self):
        self.cdll.delete_gm_to_dirac_short_double(self.gm_to_dirac_double)
        self.cdll.delete_gm_to_dirac_short_float(self.gm_to_dirac_float)
        self.cdll.delete_gm_to_dirac_short_standard_normal_deviation_double(self.gm_to_dirac_snd_double)
        self.cdll.delete_gm_to_dirac_short_standard_normal_deviation_float(self.gm_to_dirac_snd_float)

    def approximate_double(self, covDiag, L, N, bMax, x, wX, result, options) -> (bool, ctypes.Array):
        success = self.cdll.gm_to_dirac_short_double_approximate(
            self.gm_to_dirac_double,
            covDiag,
            L,
            N,
            bMax,
            x,
            wX,
            ctypes.byref(result),
            ctypes.byref(options)
        )
        return success, x
    
    def approximate_float(self, covDiag, L, N, bMax, x, wX, result, options) -> (bool, ctypes.Array):
        success = self.cdll.gm_to_dirac_short_float_approximate(
            self.gm_to_dirac_float,
            covDiag,
            L,
            N,
            bMax,
            x,
            wX,
            ctypes.byref(result),
            ctypes.byref(options)
        )
        return success, x
    
    def approximate_snd_double(self, L, N, bMax, x, wX, result, options) -> (bool, ctypes.Array):
        success = self.cdll.gm_to_dirac_short_standard_normal_deviation_double_approximate(
            self.gm_to_dirac_snd_double,
            L,
            N,
            bMax,
            x,
            wX,
            ctypes.byref(result),
            ctypes.byref(options)
        )
        return success, x
    
    def approximate_snd_float(self, L, N, bMax, x, wX, result, options) -> (bool, ctypes.Array):
        success = self.cdll.gm_to_dirac_short_standard_normal_deviation_float_approximate(
            self.gm_to_dirac_snd_float,
            L,
            N,
            bMax,
            x,
            wX,
            ctypes.byref(result),
            ctypes.byref(options)
        )
        return success, x
    