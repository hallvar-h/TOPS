from dynpssimpy.dyn_models.gen import GEN

class GEN_AGC(GEN):
    def input_list():
        return super().input_list() + ['P_agc']