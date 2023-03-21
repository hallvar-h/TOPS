from dynpssimpy.dyn_models.utils import DAEModel
from dynpssimpy.dyn_models.blocks import *
from dynpssimpy.dyn_models.utils import auto_init


class AGC1(DAEModel):
    def input_list(self):
        return ['gen_speed_dev', 'P_tie']
    
    def connections(self):
        return [
            {
                'input': 'gen_speed_dev',
                'source': {
                    'container': 'gen',
                    'mdl': '*',
                    'id': self.par['gen'],
                },
                'output': 'speed',
            },
            {
                'input': 'P_tie',
                'source': {
                    'container': 'lines',
                    'mdl': '*',
                    'id': self.par['line'],
                },
                'output': 'p_from',
            },
            {
                'output': 'output',
                'destination': {
                    'container': 'gen',
                    'mdl': '*',
                    'id': self.par['gen'],
                },
                'input': 'P_agc',
            }
        ]
    
    
    def state_list(self):
        return ['x_1']

    def state_derivatives(self, dx, x, v):
        dX = self.local_view(dx)

        # Update state variable
        dX['x_1'][:] = self.ACE(x, v) # p['K_i']*ace

    def ACE(self, x, v):
        return self.par['lambda']*self.gen_speed_dev(x, v) - (self.P_tie(x, v) - self.int_par['bias'])
    
    def delta_P_ref(self, x, v):
        X = self.local_view(x)
        p = self.par
        s_1 = p['K_p']*self.ACE(x, v)
        s_3 = p['K_i']*X['x_1']
        return s_1 + s_3

    def output(self, x, v):
        return np.minimum(np.maximum(-1.0, self.par['alpha']*self.delta_P_ref(x, v)), 1.0)

    def int_par_list(self):
        return ['bias']

    def init_from_connections(self, x0, v0, output_0):
        auto_init(self, x0, v0, output_0['output'])
        # self.int_par['bias'] = output_0['output']