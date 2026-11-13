import numpy as np
from tops.dyn_models.utils import DAEModel
from tops.dyn_models.blocks import (
    Integrator2, Gain, Limiter, TimeConstant, TimeConstantGain, TimeConstantLims, LeadLag,
    PIRegulator2Lims, Backlash,
)
from tops.dyn_models.utils import auto_init
from tops.events import ModelEvents


class GovEvents(ModelEvents):
    actions = ('set_reference',)

    def set_reference(self, gov_name, P_m):
        """Set the power reference of a governor: the mechanical power that it gives in steady state at rated speed.
        At other speeds, its droop moves the power away from the reference.

        :param gov_name: Name of the governor.
        :param P_m: Mechanical power in p.u. of the generator's rating (machine base).
        :return: No effects on the network.
        """
        try:
            value = float(P_m)
        except (TypeError, ValueError):
            value = np.nan
        if not np.isfinite(value):
            raise ValueError(f'P_m must be a finite number, got {P_m!r}.')
        self.mdl.set_reference(self._unit_idx(gov_name), value)
        return []


class GOV:
    events_class = GovEvents

    def input_list(self):
        return ['input', 'P_n_gen', 'bias']

    def reference_bias(self, P_m, idx=slice(None)):
        """The bias input that gives a mechanical power in steady state at rated speed.

        :param P_m: Mechanical power in p.u. of the generator's rating (machine base).
        :param idx: Indices of the units.
        """
        raise NotImplementedError(f'{type(self).__name__} has no power reference.')

    def set_reference(self, idx, P_m):
        """Set the power reference of units (see ``GovEvents.set_reference``).

        :param idx: Indices of the units.
        :param P_m: Mechanical power in p.u. of the generator's rating (machine base).
        """
        self.set_input('bias', self.reference_bias(P_m, idx), idx)

    def output_list(self):
        return ['output']

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
                'input': 'P_n_gen',
                'source': {
                    'container': 'gen',
                    'mdl': '*',
                    'id': self.par['gen'],
                },
                'output': 'P_nom',
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



class TGOV1(GOV, DAEModel):
    def add_blocks(self):
        p = self.par
        self.droop = Gain(K=1/p['R'])
        self.time_constant_lim = TimeConstantLims(T=p['T_1'], V_min=p['V_min'], V_max=p['V_max'])
        self.lead_lag = LeadLag(T_1=p['T_2'], T_2=p['T_3'])
        self.damping_gain = Gain(K=p['D_t'])

        self.droop.input = lambda x, v: -self.input(x, v) + self.bias(x, v)
        self.time_constant_lim.input = lambda x, v: self.droop.output(x, v)
        self.lead_lag.input = lambda x, v: self.time_constant_lim.output(x, v)
        self.damping_gain.input = lambda x, v: self.input(x, v)

        self.output = lambda x, v: self.lead_lag.output(x, v) - self.damping_gain.output(x, v)

    def reference_bias(self, P_m, idx=slice(None)):
        return self.par['R'][idx]*P_m

    def init_from_connections(self, x0, v0, output_0):
        self._input_values['bias'] = self.droop.initialize(
            x0, v0, self.time_constant_lim.initialize(
                x0, v0, self.lead_lag.initialize(x0, v0, output_0['output'])
            )
        )


class HYGOV(GOV, DAEModel):
    '''
    Implementation of the HYGOV model. Some limiters are missing.
    '''
    def add_blocks(self):
        p = self.par
        g_min = p['g_min'] if 'g_min' in p.dtype.names else np.zeros(self.n_units)
        g_max = p['g_max'] if 'g_max' in p.dtype.names else 1*np.ones(self.n_units)

        self.time_constant_1 = TimeConstant(T=p['T_f'])
        self.pi_reg = PIRegulator2Lims(T_1=p['T_r'], T_2=p['T_r']*p['r'], x_min=g_min, x_max=g_max)
        self.gain = Gain(K=p['R'])
        self.time_constant_2 = TimeConstant(T=p['T_g'])
        self.gain_A_t = Gain(K=p['A_t'])
        self.integrator = Integrator2(T=p['T_w'])

        self.time_constant_1.input = lambda x, v: -self.input(x, v) + self.bias(x, v) - p['R']*self.c(x, v)
        self.pi_reg.input = self.time_constant_1.output  # This should have a limiter
        self.c = self.pi_reg.output
        self.time_constant_2.input = self.c
        # self.g = self.time_constant_2.output
        if 'backlash' in self.par.dtype.names:
            self.backlash = Backlash(db=p['backlash'])
            self.backlash.input = self.time_constant_2.output
            self.g = self.backlash.output
        else:
            self.g = self.time_constant_2.output
            
        self.q = self.integrator.output
        self.div = lambda x, v: self.q(x, v)/self.g(x, v)
        self.h = lambda x, v: self.div(x, v)**2
        self.integrator.input = lambda x, v: -self.h(x, v) + 1
        self.gain_A_t.input = lambda x, v: (self.q(x, v) - p['q_nl'])*self.h(x, v)
        self.output = self.gain_A_t.output
    
    def reference_bias(self, P_m, idx=slice(None)):
        p = self.par
        return p['R'][idx]*(P_m/p['A_t'][idx] + p['q_nl'][idx])

    def init_from_connections(self, x0, v0, output_0):
        p = self.par
        q_p = self.gain_A_t.initialize(x0, v0, output_0['output'])
        q = q_p + p['q_nl']
        self.integrator.initialize(x0, v0, q)
        c = g = q
        if 'backlash' in self.par.dtype.names:
            self.backlash.initialize(x0, v0, q)
        self.time_constant_2.initialize(x0, v0, q)
        self.pi_reg.initialize(x0, v0, q)
        self.pi_reg.output(x0, v0)
        self.time_constant_1.initialize(x0, v0, q*0)
        self._input_values['bias'] = p['R']*c
        


class IEESGO(GOV, DAEModel):
    def add_blocks(self):
        p = self.par
        self.lead_lag = LeadLag(T_1=p['T_2'], T_2=p['T_1'])
        self.time_constant_gain_k1_t3 = TimeConstantGain(K=p['K_1'], T=p['T_3'])
        self.limiter = Limiter(Min=p['P_min'], Max=p['P_max'])
        self.time_constant_2 = TimeConstant(T=p['T_4'])

        self.gain_1_minus_k2 = Gain(K=1-p['K_2'])
        self.time_constant_gain_k2_t5 = TimeConstantGain(K=p['K_2'], T=p['T_5'])
        self.gain_1_minus_k3 = Gain(K=1-p['K_3'])
        self.time_constant_gain_k3_t6 = TimeConstantGain(K=p['K_3'], T=p['T_6'])

        self.lead_lag.input = lambda x, v: self.input(x, v)
        self.time_constant_gain_k1_t3.input = self.lead_lag.output
        self.limiter.input = lambda x, v: -self.time_constant_gain_k1_t3.output(x, v) + self.bias(x, v)

        self.time_constant_2.input = self.limiter.output
        self.gain_1_minus_k2.input = self.time_constant_2.output
        self.time_constant_gain_k2_t5.input = self.time_constant_2.output
        self.gain_1_minus_k3.input = self.time_constant_gain_k2_t5.output
        self.time_constant_gain_k3_t6.input = self.time_constant_gain_k2_t5.output

        def output_sum(x, v):
            return self.gain_1_minus_k2.output(x, v) + self.gain_1_minus_k3.output(x, v) + self.time_constant_gain_k3_t6.output(x, v)
        
        P_n = p['P_N']
        self.output = lambda x, v: output_sum(x, v)*(P_n/self.P_n_gen(x, v) + 1*(P_n==0))
    
    def reference_bias(self, P_m, idx=slice(None)):
        P_N = self.par['P_N'][idx]
        scale = np.where(P_N == 0, 1.0, self._P_n_gen_0[idx]/np.where(P_N == 0, 1.0, P_N))
        return P_m*scale

    def init_from_connections(self, x0, v0, output_0):
        auto_init(self, x0, v0, output_0['output'], free_inputs=['bias'])
        self._P_n_gen_0 = np.array(self.P_n_gen(x0, v0), dtype=float)*np.ones(self.n_units)