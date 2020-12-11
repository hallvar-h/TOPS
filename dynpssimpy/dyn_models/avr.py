import numpy as np


class SEXS:
    def __init__(self):
        self.state_list = ['x', 'e_f']
        self.int_par_list = ['x_bias']

    @staticmethod
    def initialize(x_0, output_0, p, int_par):
        bias = 1 / p['K'] * output_0
        int_par['x_bias'] = bias
        x_0['x'][:] = (p['T_a'] - p['T_b']) * bias
        x_0['e_f'][:] = output_0

    @staticmethod
    def _update(dx, x, input, p, int_par):

        u = input + int_par['x_bias']
        v_1 = 1 / p['T_b'] * (p['T_a'] * u - x['x'])

        dx['x'][:] = v_1 - u
        dx['e_f'][:] = 1/p['T_e'] * (p['K'] * v_1 - x['e_f'])

        # Lims on state variable e_f (clamping)
        lower_lim_idx = (x['e_f'] <= p['E_min']) & (dx['e_f'] < 0)
        dx['e_f'][lower_lim_idx] *= 0

        upper_lim_idx = (x['e_f'] >= p['E_max']) & (dx['e_f'] > 0)
        dx['e_f'][upper_lim_idx] *= 0

        output = np.minimum(np.maximum(x['e_f'], p['E_min']), p['E_max'])

        return output


if __name__ == '__main__':
    # Simple speed test of model (jit vs nojit)
    import time
    from numba import jit

    avr = SEXS()
    n = 20
    avr.par = {
        'E_max': np.array([3.]*n),
        'E_min': np.array([-3.]*n),
        'K': np.array([100.]*n),
        'T_a': np.array([2.]*n),
        'T_b': np.array([10.]*n),
        'T_e': np.array([0.5]*n),
        'gen': np.array(['G1']*n, dtype='<U2'),
        'name': np.array(['AVR1']*n, dtype='<U4')
    }
    avr.state_idx = {
        'e_f': np.arange(n, 2*n),
        'x': np.arange(n)
    }
    avr.int_par = {
        'x_bias': np.array([0]*n)
    }

    # x = np.zeros(2*n)
    x = np.arange(2*n)
    dx = np.arange(2*n)
    # update_jit = jit()(avr._update)
    # update_jit(dx, x, 1, avr.par, avr.state_idx, avr.int_par)

    n_it = 1000
    t_0 = time.time()
    for _ in range(n_it):
        avr._update(dx, x, 1, avr.par, avr.state_idx, avr.int_par)
    print(time.time() - t_0)

    # t_0 = time.time()
    # for _ in range(n_it):
    #     update_jit(dx, x, 1, avr.par, avr.state_idx, avr.int_par)
    # print(time.time() - t_0)