import numpy as np


class TGOV1:
    def __init__(self):
        self.state_list = ['x_1', 'x_2']
        self.int_par_list = ['x_1_bias']
        self.input_list = ['speed_dev']
        self.output_list = ['P_m']

    @staticmethod
    def initialize(x_0, input, output, p, int_par):
        v_2 = np.minimum(np.maximum(output['P_m'], p['V_min']), p['V_max'])
        v_1 = v_2
        v_3 = v_2

        int_par['x_1_bias'] = p['R'] * v_1

        x_0['x_1'][:] = v_2
        x_0['x_2'][:] = p['T_2'] * v_2 - p['T_3'] * v_3

    @staticmethod
    def _update(dx, x, input, output, p, int_par):

        speed_dev = input['speed_dev']
        v_1 = 1 / p['R'] * (speed_dev + int_par['x_1_bias'])
        v_2 = np.minimum(np.maximum(x['x_1'], p['V_min']), p['V_max'])
        v_3 = p['T_2'] / p['T_3'] * v_2 - 1 / p['T_3'] * x['x_2']
        delta_p_m = v_3 - p['D_t'] * speed_dev

        output['P_m'][:] = delta_p_m

        dx['x_1'][:] = 1 / p['T_1'] * (v_1 - v_2)
        dx['x_2'][:] = v_3 - v_2

        # Lims on state variable x_1 (clamping)
        lower_lim_idx = (x['x_1'] <= p['V_min']) & (dx['x_1'] < 0)
        dx['x_1'][lower_lim_idx] *= 0

        upper_lim_idx = (x['x_1'] >= p['V_max']) & (dx['x_1'] > 0)
        dx['x_1'][upper_lim_idx] *= 0
