from dynpssimpy.dyn_models.blocks import *
from dynpssimpy.dyn_models.utils import auto_init


class GOV:
    def connections(self):
        return [
            {
                'input': 'input',
                'source': {
                    'container': 'gen',
                    'mdl': '*',
                    'id': self.par['gen'],
                },
                'output': 'speed',
            },
            {
                'output': 'output',
                'destination': {
                    'container': 'gen',
                    'mdl': '*',
                    'id': self.par['gen'],
                },
                'input': 'P_m',
            }
        ]



class TGOV1(DAEModel, GOV):
    def add_blocks(self):
        p = self.par
        self.droop = Gain(K=1/p['R'])
        self.time_constant_lim = TimeConstantLims(T=p['T_1'], V_min=p['V_min'], V_max=p['V_max'])
        self.lead_lag = LeadLag(T_1=p['T_2'], T_2=p['T_3'])
        self.damping_gain = Gain(K=p['D_t'])

        self.droop.input = lambda x, v: -self.input(x, v) + self.int_par['bias']
        self.time_constant_lim.input = lambda x, v: self.droop.output(x, v)
        self.lead_lag.input = lambda x, v: self.time_constant_lim.output(x, v)
        self.damping_gain.input = lambda x, v: self.input(x, v)

        self.output = lambda x, v: self.lead_lag.output(x, v) - self.damping_gain.output(x, v)

    def int_par_list(self):
        return ['bias']

    def init_from_connections(self, x0, v0, output_0):
        p = self.par
        self.int_par['bias'] = self.droop.initialize(
            x0, v0, self.time_constant_lim.initialize(
                x0, v0, self.lead_lag.initialize(x0, v0, output_0['output'])
            )
        )


class HYGOV(DAEModel, GOV):
    def int_par_list(self):
        return ['bias']

    def add_blocks(self):
        p = self.par
        self.time_constant_1 = TimeConstant(T=p['T_f'])
        self.pi_reg = PIRegulator2(T_1=p['T_r'], T_2=p['T_r']*p['r'])
        self.gain = Gain(K=p['R'])
        self.time_constant_2 = TimeConstant(T=p['T_g'])
        self.gain_A_t = Gain(K=p['A_t'])
        self.integrator = Integrator2(T=p['T_w'])
        
        self.time_constant_1.input = lambda x, v: -self.input(x, v) + self.int_par['bias'] - p['R']*self.c(x, v)
        self.pi_reg.input = self.time_constant_1.output  # This should have a limiter
        self.c = self.pi_reg.output
        self.time_constant_2.input = self.c
        self.g = self.time_constant_2.output
        self.q = self.integrator.output
        self.div = lambda x, v: self.q(x, v)/self.time_constant_2.output(x, v)
        self.h = lambda x, v: self.div(x, v)**2
        self.integrator.input = lambda x, v: -self.h(x, v) + 1
        self.gain_A_t.input = lambda x, v: (self.q(x, v) - p['q_nl'])*self.h(x, v)
        self.output = self.gain_A_t.output
        
    def int_par_list(self):
        return ['bias']
    
    def init_from_connections(self, x0, v0, output_0):
        auto_init(self, x0, v0, output_0['output'])
        # input_0 = self.auto_init(x0, v0, output_0)
        # p = self.par
        # self.int_par['bias'] = self.droop.initialize(
        #     x0, v0, self.time_constant_lim.initialize(
        #         x0, v0, self.lead_lag.initialize(x0, v0, output_0['output'])
        #     )
        # )