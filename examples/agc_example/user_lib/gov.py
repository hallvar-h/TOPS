from dynpssimpy.dyn_models.gov import TGOV1 as TGOV1_0
from dynpssimpy.dyn_models.gov import HYGOV as HYGOV_0

class GOV_AGC:
    def connections(self):
        return super().connections() + [
            {
                'input': 'P_agc',
                'source': {
                    'container': 'gen',
                    'mdl': '*',
                    'id': self.par['gen'],
                },
                'output': 'P_agc',
            }
        ]
    
    def input_list(self):
        return super().input_list() + ['P_agc']
    

class TGOV1(GOV_AGC, TGOV1_0):    
    def add_blocks(self):
        super().add_blocks()
        self.droop.input = lambda x, v: -self.input(x, v) + self.P_agc(x, v) + self.int_par['bias']

class HYGOV(GOV_AGC, HYGOV_0):
    def add_blocks(self):
        super().add_blocks()
        self.time_constant_1.input = lambda x, v: -self.input(x, v) + self.int_par['bias'] - p['R']*self.c(x, v) + self.P_agc(x, v)