from dynpssimpy.dyn_models.utils import DAEModel
from dynpssimpy.dyn_models.gov import GOV
from dynpssimpy.dyn_models.blocks import *
from dynpssimpy.dyn_models.utils import auto_init


class MYGOV(GOV, DAEModel):
    def add_blocks(self):
        p = self.par
        self.integrator_Kw = IntegratorK(K=p['Kw'])
        self.integrator_K = IntegratorK(K=p['K'])
        self.droop_R = Gain(K=p['R'])

        self.integrator_Kw.input = lambda x, v: self.input(x, v)
        self.droop_R.input = lambda x, v: self.integrator_K.output(x, v) - self.integrator_Kw.output(x, v) - self.int_par['bias']
        self.integrator_K.input = lambda x, v: self.input(x, v) - self.droop_R.output(x, v)
        
        self.output = self.integrator_K.output

    def int_par_list(self):
        return ['bias']

    def init_from_connections(self, x0, v0, output_0):
        auto_init(self, x0, v0, output_0['output'])
        # self.int_par['bias'] = output_0['output']
        # self.integrator_K.initialize(x0, v0, output_0['output'])
        # self.integrator_Kw.initialize(x0, v0, output_0['output']*0)


class MYGOV2(GOV, DAEModel):
    '''Same as MYGOV, but without using blocks'''
    def int_par_list(self):
        return ['P0']  # , 'wref']
    
    def state_list(self):
        return ['x_1', 'x_2']

    def init_from_connections(self, x0, v0, output_0):
        # auto_init(self, x0, v0, output_0)
        X0 = self.local_view(x0)
        X0['x_1'] = output_0['output']
        X0['x_2'] = 0
        self.int_par['P0'] = output_0['output']

    def state_derivatives(self, dx, x, v):
        dX = self.local_view(dx)
        X = self.local_view(x)
        p = self.par

        speed_dev =  self.input(x, v)  # + self.int_par['wref']
        dP = X['x_1'] - self.int_par['P0'] - X['x_2']

        dX['x_1'] = (speed_dev - p['R']*dP)*p['K']
        dX['x_2'] = speed_dev*p['Kw']
        # output['P_m'][:] = x['x_1']

    def output(self, x, v):
        return self.local_view(x)['x_1']