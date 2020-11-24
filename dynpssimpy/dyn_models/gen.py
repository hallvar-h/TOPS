import numpy as np


class GEN:
    def __init__(self):
        self.state_list = ['speed', 'angle', 'e_q_t', 'e_d_t', 'e_q_st', 'e_d_st']

    def initialize(self, v_0, S_0):
        p = self.par
        s = self.state_idx

        v_g = v_0
        S_g = S_0
        P_m_0 = S_0.real/p['PF_n']

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

        x_0 = np.concatenate((
            speed,
            angle,
            e_q_t,
            e_d_t,
            e_q_st,
            e_d_st,
        ))

        inputs_0 = e_q_0, P_m_0

        return x_0, inputs_0

    @staticmethod
    def _current_injections(x, p, s):
        d = np.exp(1j * (x[s['angle']] - np.pi / 2))
        q = np.exp(1j * x[s['angle']])
        i_inj_d = x[s['e_q_st']] / (1j * p['X_d_st']) * q * p['N_par']
        i_inj_q = x[s['e_d_st']] / (1j * p['X_q_st']) * d * p['N_par']
        return i_inj_d, i_inj_q

    @staticmethod
    def _update(dx, f, x, v_g, e_q, P_m, v_aux, p, s):

        d = np.exp(1j * (x[s['angle']] - np.pi / 2))
        q = np.exp(1j * x[s['angle']])

        e_st = x[s['e_q_st']] * q + x[s['e_d_st']] * d
        e_t = x[s['e_q_t']] * q + x[s['e_d_t']] * d
        e = e_q * q

        I_g = (e_st - v_g) / (1j * p['X_d_st'])

        v_g_dq = v_g * np.exp(1j * (np.pi / 2 - x[s['angle']]))
        v_d = v_g_dq.real
        v_q = v_g_dq.imag

        I_g_dq = I_g * np.exp(1j * (np.pi / 2 - x[s['angle']]))
        I_d = I_g_dq.real
        I_q = I_g_dq.imag

        e_q_tmp = v_g + 1j * p['X_q'] * I_g

        P_e = (x[s['e_q_st']] * I_q + x[s['e_d_st']] * I_d)/p['PF_n']  # - (x_d_st - x_q_st) * i_d * i_q

        # Generators
        T_m = P_m / (1 + x[s['speed']])

        H = p['H']/p['PF_n']

        dx[:] = np.concatenate((
            1/(2*H)*(T_m - P_e - p['D'] * x[s['speed']]),
            x[s['speed']]*2*np.pi*f,
            1/(p['T_d0_t'])*(e_q + v_aux - x[s['e_q_t']] - I_d * (p['X_d'] - p['X_d_t'])),
            1/(p['T_q0_t'])*(-x[s['e_d_t']] + I_q * (p['X_q'] - p['X_q_t'])),
            1/(p['T_d0_st']) * (x[s['e_q_t']] - x[s['e_q_st']] - I_d * (p['X_d_t'] - p['X_d_st'])),
            1/(p['T_q0_st']) * (x[s['e_d_t']] - x[s['e_d_st']] + I_q * (p['X_q_t'] - p['X_q_st'])),
        ))

        return

