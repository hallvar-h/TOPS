from dynpssimpy.dyn_models.blocks import *
from dynpssimpy.dyn_models.utils import auto_init


class AVR:
    def connections(self):
        return [
            {
                'input': 'v_t',
                'source': {
                    'container': 'gen',
                    'mdl': '*',
                    'id': self.par['gen'],
                },
                'output': 'v_t_abs',
            },
            {
                'input': 'v_setp',
                'source': {
                    'container': 'gen',
                    'mdl': '*',
                    'id': self.par['gen'],
                },
                'output': 'v_setp',
            },
            {
                'input': 'v_pss',
                'source': {
                    'container': 'gen',
                    'mdl': '*',
                    'id': self.par['gen'],
                },
                'output': 'v_pss',
            },
            {
                'output': 'output',
                'destination': {
                    'container': 'gen',
                    'mdl': '*',
                    'id': self.par['gen'],
                },
                'input': 'E_f',
            }
        ]


class SEXS(DAEModel, AVR):
    def input_list(self):
        return ['v_setp', 'v_t', 'v_pss']

    def add_blocks(self):
        p = self.par
        self.tg_red = LeadLag(T_1=p['T_a'], T_2=p['T_b'])
        self.gain = Gain(K=p['K'])
        self.time_constant_lim = TimeConstantLims(T=p['T_e'], V_min=p['E_min'], V_max=p['E_max'])
        self.tg_red.input = lambda x, v: self.v_setp(x, v) - self.v_t(x, v) + self.v_pss(x, v) + self.int_par['bias']
        self.gain.input = lambda x, v: self.tg_red.output(x, v)
        self.time_constant_lim.input = lambda x, v: self.gain.output(x, v)
        self.output = lambda x, v: self.time_constant_lim.output(x, v)

    def int_par_list(self):
        return ['bias']

    def init_from_connections(self, x0, v0, output_0):
        # auto_init(self, x0, v0, output_0['output'])
        self.int_par['bias'] = \
            self.tg_red.initialize(
                x0, v0, self.gain.initialize(
                    x0, v0, self.time_constant_lim.initialize(
                        x0, v0, output_0['output'])
            )
        )



class SEXS_PI(DAEModel, AVR):
    def input_list(self):
        return ['v_setp', 'v_t', 'v_pss']

    def add_blocks(self):
        p = self.par
        self.pi_regulator = PIRegulator(K_p=p['K_p'], K_i=p['K_i'])
        self.tg_red = LeadLag(T_1=p['T_a'], T_2=p['T_b'])
        self.gain = Gain(K=p['K'])
        self.time_constant_lim = TimeConstantLims(T=p['T_e'], V_min=p['E_min'], V_max=p['E_max'])

        self.v_setp_lag = TimeConstant(T=p['T_ext'])
        self.v_setp_lag.input = lambda x, v: self.v_setp(x, v)
        
        self.pi_regulator.input = lambda x, v: self.v_setp_lag.output(x, v) - self.v_t(x, v) + self.v_pss(x, v)  # + self.int_par['bias']
        self.tg_red.input = lambda x, v: self.pi_regulator.output(x, v)
        self.gain.input = lambda x, v: self.tg_red.output(x, v)
        self.time_constant_lim.input = lambda x, v: self.gain.output(x, v)
        self.output = lambda x, v: self.time_constant_lim.output(x, v)

    # def int_par_list(self):
    #     return ['bias']

    def init_from_connections(self, x0, v0, output_0):
        # v_0 = np.ones(self.n_units)
        v_0 = self.v_setp(x0, v0)
        self.v_setp_lag.initialize(x0, v0, v_0)
        self.pi_regulator.initialize(
            x0, v0, self.tg_red.initialize(
                x0, v0, self.gain.initialize(
                    x0, v0, self.time_constant_lim.initialize(
                        x0, v0, output_0['output']
                    )
                )
            )
        )

