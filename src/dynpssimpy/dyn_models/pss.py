from dynpssimpy.dyn_models.blocks import *


class PSS:
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
                'input': 'v_pss',
            }
        ]


class STAB1(DAEModel, PSS):
    def add_blocks(self):
        p = self.par
        self.gain = Gain(K=p['K'])  # , input=self.input)
        self.washout = Washout(T_w=p['T'])  # , input=self.input)
        self.lead_lag_1 = LeadLag(T_1=p['T_1'], T_2=p['T_3'])  # , input=self.washout.output)
        self.lead_lag_2 = LeadLag(T_1=p['T_2'], T_2=p['T_4'])  # , input=self.lead_lag_1.output)
        self.limiter = Limiter(Min=-p['H_lim'], Max=p['H_lim'])  # , input=self.lead_lag_1.output)

        self.gain.input = lambda x, v: self.input(x, v)
        self.washout.input = lambda x, v: self.gain.output(x, v)
        self.lead_lag_1.input = lambda x, v: self.washout.output(x, v)
        self.lead_lag_2.input = lambda x, v: self.lead_lag_1.output(x, v)
        self.limiter.input = lambda x, v: self.lead_lag_2.output(x, v)

        self.output = lambda x, v: self.limiter.output(x, v)