import numpy as np


class GEN:
    def __init__(self):
        self.state_list = ['speed', 'angle', 'e_q_t', 'e_d_t', 'e_q_st', 'e_d_st']
        self.input_list = ['V_t_abs', 'V_t_angle', 'P_m', 'E_f', 'v_aux', 'v_pss']
        self.int_par_list = ['f']
        self.output_list = ['P_e', 'Q', 'T_m']

    @staticmethod
    def initialize(x_0, input, output, p, int_par):

        # Converting to terminal voltage phasor (inputs are only float, not complex).
        # This is to avoid having to specify units in input_list in model definition.
        v_g = input['V_t_abs']*np.exp(1j*input['V_t_angle'])  # Potential p.u. error
        S_g = output['P_e'] + 1j*output['Q']
        P_m_0 = S_g.real/p['PF_n']

        I_g = np.conj(S_g / v_g)

        e_q_tmp = v_g + 1j * p['X_q'] * I_g
        angle = np.angle(e_q_tmp)
        speed = np.zeros_like(angle)

        d = np.exp(1j * (angle - np.pi / 2))
        q = np.exp(1j * angle)

        I_g_dq = I_g * np.exp(1j * (np.pi / 2 - angle))
        I_d = I_g_dq.real
        I_q = I_g_dq.imag  # q-axis leading d-axis

        v_g_dq = v_g * np.exp(1j * (np.pi / 2 - angle))
        v_d = v_g_dq.real
        v_q = v_g_dq.imag

        e_q_t = v_q + p['X_d_t'] * I_d
        e_d_t = v_d - p['X_q_t'] * I_q
        e_t = e_q_t * q + e_d_t * d

        e_q_st = v_q + p['X_d_st'] * I_d
        e_d_st = v_d - p['X_q_st'] * I_q
        e_st = e_q_st * q + e_d_st * d

        e_q = e_q_t + I_d * (p['X_d'] - p['X_d_t'])
        e = e_q * np.exp(1j * angle)
        e_q_0 = e_q.copy()

        P_m = P_m_0.copy()
        P_e = e_q_st * I_q + e_d_st * I_d - (p['X_d_st'] - p['X_q_st']) * I_d * I_q

        x_0['speed'][:] = speed
        x_0['angle'][:] = angle
        x_0['e_q_t'][:] = e_q_t
        x_0['e_d_t'][:] = e_d_t
        x_0['e_q_st'][:] = e_q_st
        x_0['e_d_st'][:] = e_d_st

        input['E_f'][:] = e_q_0
        input['P_m'][:] = P_m_0

    @staticmethod
    def _current_injections(x, p):
        d = np.exp(1j * (x['angle'] - np.pi / 2))
        q = np.exp(1j * x['angle'])
        i_inj_d = x['e_q_st'] / (1j * p['X_d_st']) * q * p['N_par']
        i_inj_q = x['e_d_st'] / (1j * p['X_q_st']) * d * p['N_par']
        return i_inj_d, i_inj_q

    @staticmethod
    def _update(dx, x, input, output, p, int_par):

        # Converting to terminal voltage phasor (inputs are only float, not complex).
        # This is to avoid having to specify units in input_list in model definition.
        V_g = input['V_t_abs']*np.exp(1j*input['V_t_angle'])


        d = np.exp(1j * (x['angle'] - np.pi / 2))
        q = np.exp(1j * x['angle'])

        e_st = x['e_q_st'] * q + x['e_d_st'] * d
        e_t = x['e_q_t'] * q + x['e_d_t'] * d
        # e = input['e_q'] * q

        I_g = (e_st - V_g) / (1j * p['X_d_st'])

        v_g_dq = V_g * np.exp(1j * (np.pi / 2 - x['angle']))
        v_d = v_g_dq.real
        v_q = v_g_dq.imag

        I_g_dq = I_g * np.exp(1j * (np.pi / 2 - x['angle']))
        I_d = I_g_dq.real
        I_q = I_g_dq.imag

        e_q_tmp = V_g + 1j * p['X_q'] * I_g

        P_e = (x['e_q_st'] * I_q + x['e_d_st'] * I_d)/p['PF_n']  # - (x_d_st - x_q_st) * i_d * i_q
        output['P_e'][:] = P_e
        # output['Q'][:]

        # Generators
        T_m = input['P_m'] / (1 + x['speed'])
        output['T_m'][:] = T_m

        H = p['H']/p['PF_n']

        dx['speed'][:] = 1/(2*H)*(T_m - P_e - p['D'] * x['speed'])
        dx['angle'][:] = x['speed']*2*np.pi*int_par['f']
        dx['e_q_t'][:] = 1/(p['T_d0_t'])*(input['E_f'] + input['v_aux'] - x['e_q_t'] - I_d * (p['X_d'] - p['X_d_t']))
        dx['e_d_t'][:] = 1/(p['T_q0_t'])*(-x['e_d_t'] + I_q * (p['X_q'] - p['X_q_t']))
        dx['e_q_st'][:] = 1/(p['T_d0_st']) * (x['e_q_t'] - x['e_q_st'] - I_d * (p['X_d_t'] - p['X_d_st']))
        dx['e_d_st'][:] = 1/(p['T_q0_st']) * (x['e_d_t'] - x['e_d_st'] + I_q * (p['X_q_t'] - p['X_q_st']))


class GEN2(GEN):
    pass