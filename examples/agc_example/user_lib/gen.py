from dynpssimpy.dyn_models.gen import GEN as GEN_0

class GEN(GEN_0):
    def input_list(self):
        return super().input_list() + ['P_agc']