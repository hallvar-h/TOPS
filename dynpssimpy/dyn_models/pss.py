import numpy as np


class STAB1:
    def __init__(self):
        self.state_list = ['x_1', 'x_2', 'x_3']
        self.int_par_list = []

    @staticmethod
    def _update(dx, x, input, p, int_par):
        u = input

        v_1 = (p['K'] * u - x['x_1']) / p['T']
        v_2 = 1 / p['T_3'] * (p['T_1'] * v_1 - x['x_2'])
        v_3 = 1 / p['T_4'] * (p['T_2'] * v_2 - x['x_3'])

        output = np.minimum(np.maximum(v_3, -p['H_lim']), p['H_lim'])

        dx['x_1'][:] = v_1
        dx['x_2'][:] = (p['T_1'] / p['T_3'] - 1) * v_1 - 1 / p['T_3'] * x['x_2']
        dx['x_3'][:] = (p['T_2'] / p['T_4'] - 1) * v_2 - 1 / p['T_4'] * x['x_3']

        return output