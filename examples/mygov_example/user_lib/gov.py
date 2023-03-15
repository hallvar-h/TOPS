from dynpssimpy.dyn_models.utils import DAEModel
from dynpssimpy.dyn_models.gov import GOV
from dynpssimpy.dyn_models.blocks import *
from dynpssimpy.dyn_models.utils import auto_init


class MYGOV(GOV, DAEModel):
    def add_blocks(self):
        p = self.par
        self.integrator_Kw = IntegratorK(K=p['K_w'])
        self.integrator_K = IntegratorK(K=p['K'])
        self.droop_R = IntegratorK(K=p['R'])

        self.integrator_Kw.input = lambda x, v: self.input(x, v)
        self.droop_R.input = lambda x, v: self.integrator_Kw.output(x, v) - self.integrator_K.output(x, v) - self.int_par['bias']
        self.integrator_K.input = lambda x, v: self.input(x, v) - self.droop_R.output(x, v)
        
        self.output = self.integrator_K.output

    def int_par_list(self):
        return ['bias']

    def init_from_connections(self, x0, v0, output_0):
        auto_init(self, x0, v0, output_0['output'])