import numpy as np


class STAB1:
    def __init__(self):
        self.state_list = ['x_1', 'x_2', 'x_3']
        self.int_par_list = []

    def update(self, x, input):
        p = self.par
        s = self.state_idx
        u = input

        v_1 = (p['K'] * u - x[s['x_1']]) / p['T']
        v_2 = 1 / p['T_3'] * (p['T_1'] * v_1 - x[s['x_2']])
        v_3 = 1 / p['T_4'] * (p['T_2'] * v_2 - x[s['x_3']])

        output = np.minimum(np.maximum(v_3, -p['H_lim']), p['H_lim'])

        dx = np.concatenate([
            v_1,
            (p['T_1'] / p['T_3'] - 1) * v_1 - 1 / p['T_3'] * x[s['x_2']],
            (p['T_2'] / p['T_4'] - 1) * v_2 - 1 / p['T_4'] * x[s['x_3']],
        ])

        return dx, output