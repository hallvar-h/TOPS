import numpy as np


class TGOV1:
    def __init__(self):
        self.state_list = ['x_1', 'x_2']
        self.int_par_list = ['x_1_bias']

    def initialize(self, output_0):
        p = self.par
        s = self.state_idx

        v_2 = np.minimum(np.maximum(output_0, p['V_min']), p['V_max'])
        v_1 = v_2
        v_3 = v_2

        self.int_par['x_1_bias'] = p['R'] * v_1
        return np.concatenate([
            v_2,
            p['T_2'] * v_2 - p['T_3'] * v_3,
        ])

    @staticmethod
    def _update(dx, x, input, p, s, int_par):

        speed_dev = input
        v_1 = 1 / p['R'] * (speed_dev + int_par['x_1_bias'])
        v_2 = np.minimum(np.maximum(x[s['x_1']], p['V_min']), p['V_max'])
        v_3 = p['T_2'] / p['T_3'] * v_2 - 1 / p['T_3'] * x[s['x_2']]
        delta_p_m = v_3 - p['D_t'] * speed_dev

        output = delta_p_m

        dx[:] = np.concatenate((
            1 / p['T_1'] * (v_1 - v_2),
            v_3 - v_2
        ))

        # Lims on state variable x_1 (clamping)
        lower_lim_idx = (x[s['x_1']] <= p['V_min']) & (dx[s['x_1']] < 0)
        dx[s['x_1'][lower_lim_idx]] *= 0

        upper_lim_idx = (x[s['x_1']] >= p['V_max']) & (dx[s['x_1']] > 0)
        dx[s['x_1'][upper_lim_idx]] *= 0

        return output
