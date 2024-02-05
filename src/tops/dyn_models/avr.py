from tops.dyn_models.blocks import *
from tops.dyn_models.utils import auto_init


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


class IEEET1(DAEModel, AVR):
    '''
    Saturation not yet included!
    '''
    def input_list(self):
        return ['v_setp', 'v_t', 'v_pss']

    def add_blocks(self):
        p = self.par
        self.time_constant_Tr = TimeConstant(T=p['T_r'])
        self.time_constant_gain_Ka_Ta = TimeConstantGainLims(K=p['K_a'], T=p['T_a'], V_min=p['V_rmin'], V_max=p['V_rmax'])
        self.time_constant_gain_Ke_Te = TimeConstantVar(K=p['K_e'], T=p['T_e'])
        # self.saturation = Saturation(E_1=p['E_1'], S_e1=p['S_e1'], E_2=p['E_2'], S_e2=p['S_e2'])
        self.diff_Kf_Tf = WashoutGain(K=p['K_f'], T_w=p['T_f'])
        
        self.time_constant_Tr.input = lambda x, v: self.v_t(x, v)
        self.v_error = lambda x, v: self.v_setp(x, v) - self.time_constant_Tr.output(x, v) + self.v_pss(x, v) + self.int_par['bias']
        self.time_constant_gain_Ka_Ta.input = lambda x, v: self.v_error(x, v) - self.diff_Kf_Tf.output(x, v)
        self.time_constant_gain_Ke_Te.input = lambda x, v: self.time_constant_gain_Ka_Ta.output(x, v) # + self.saturation.output(x, v)
        self.diff_Kf_Tf.input = self.time_constant_gain_Ke_Te.output
        # self.saturation.input = lambda x, v: self.time_constant_gain_Ke_Te.output(x, v)*0
        
        self.output = self.time_constant_gain_Ke_Te.output

    def int_par_list(self):
        return ['bias']

    def init_from_connections(self, x0, v0, output_0):
        auto_init(self, x0, v0, output_0['output'])


class SCRX(DAEModel, AVR):
    '''
    Missing negative current logic!
    '''
    def input_list(self):
        return ['v_setp', 'v_t', 'v_pss']

    def add_blocks(self):
        p = self.par
        
        self.lead_lag_Tb_Ta = LeadLag(T_1=p['T_a'], T_2=p['T_b'])
        self.time_constant_gain_K_Te = TimeConstantGainLims(K=p['K'], T=p['T_e'], V_min=p['E_min'], V_max=p['E_max'])
        
        self.v_error = lambda x, v: self.v_setp(x, v) - self.v_t(x, v) + self.v_pss(x, v) + self.int_par['bias']
        self.lead_lag_Tb_Ta.input = self.v_error
        self.time_constant_gain_K_Te.input = self.lead_lag_Tb_Ta.output

        def current_logic_block(x, v):
            bus_fed_idx = p['C_switch'] == 0
            out_signal = self.time_constant_gain_K_Te.output(x, v)
            out_signal[bus_fed_idx] *= self.v_t(x, v)[bus_fed_idx]
            return out_signal            
        
        self.output = current_logic_block

    def int_par_list(self):
        return ['bias']

    def init_from_connections(self, x0, v0, output_0):
        auto_init(self, x0, v0, output_0['output'])