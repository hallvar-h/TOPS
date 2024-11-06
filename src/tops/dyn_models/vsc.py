from tops.dyn_models.blocks import *
from .pll import PLL1

class VSC(DAEModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.bus_idx = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])
        self.bus_idx_red = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])

    def bus_ref_spec(self):
        return {'terminal': self.par['bus']}

    def add_blocks(self):
        p = self.par
        self.pll = PLL1(T_filter=self.par['T_pll'], bus=p['bus'])

        self.pi_p = PIRegulator(K_p=p['P_K_p'], K_i=p['P_K_i'])
        self.pi_p.input = lambda x, v: self.P_setp(x, v) - self.P(x, v)

        self.pi_q = PIRegulator(K_p=p['Q_K_p'], K_i=p['Q_K_i'])
        self.pi_q.input = lambda x, v: self.Q_setp(x, v) - self.Q(x, v)

        self.lag_p = TimeConstant(T=p['T_i'])
        self.lag_p.input = self.pi_p.output
        self.lag_q = TimeConstant(T=p['T_i'])
        self.lag_q.input = self.pi_q.output

        self.I_d = self.lag_p.output
        self.I_q = self.lag_q.output

    def I_inj(self, x, v):
        return (self.I_d(x, v) - 1j*self.I_q(x, v))*np.exp(1j*self.pll.output(x, v))

    def input_list(self):
        return ['P_setp', 'Q_setp']

    def P(self, x, v):
        v_n = self.sys_par['bus_v_n'][self.bus_idx_red['terminal']]
        V = abs(v[self.bus_idx_red['terminal']])*v_n
        return np.sqrt(3)*V*self.I_d(x, v)

    def Q(self, x, v):
        v_n = self.sys_par['bus_v_n'][self.bus_idx_red['terminal']]
        V = abs(v[self.bus_idx_red['terminal']])*v_n
        return np.sqrt(3)*V*self.I_q(x, v)

    def load_flow_pq(self):
        return self.bus_idx['terminal'], -self.par['P_setp'], -self.par['Q_setp']

    def init_from_load_flow(self, x_0, v_0, S):
        self._input_values['P_setp'] = self.par['P_setp']
        self._input_values['Q_setp'] = self.par['Q_setp']

        v_n = self.sys_par['bus_v_n'][self.bus_idx_red['terminal']]

        V_0 = v_0[self.bus_idx_red['terminal']]*v_n

        I_d_0 = self.par['P_setp']/(abs(V_0)*np.sqrt(3))
        I_q_0 = self.par['Q_setp']/(abs(V_0)*np.sqrt(3))

        self.pi_p.initialize(
            x_0, v_0, self.lag_p.initialize(x_0, v_0, I_d_0)
        )

        self.pi_q.initialize(
            x_0, v_0, self.lag_q.initialize(x_0, v_0, I_q_0)
        )

    def current_injections(self, x, v):
        i_n = self.sys_par['s_n'] / (np.sqrt(3) * self.sys_par['bus_v_n'])
        # self.P(x, v)
        return self.bus_idx_red['terminal'], self.I_inj(x, v)/i_n[self.bus_idx_red['terminal']]


class VSC_PV(DAEModel):
    """
    'vsc': {
            'VSC_PV': [
                ['name', 'bus', 'S_n', 'P_setp', 'V_setp', 'k_p', 'k_v', 'T_p', 'T_v', 'k_pll', 'T_pll', 'T_i', 'i_max'],
                ['VSC1', 'B1',    50,   300,    0.93,    1,      1,   0.1,   0.1,     5,      1,      0.01,    1.2],
            ],
        }
    """
      

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.bus_idx = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])
        self.bus_idx_red = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])  

    def load_flow_pv(self):  
        return self.bus_idx['terminal'], -self.par['P_setp'], self.par['V_setp']

    def bus_ref_spec(self):
        return {'terminal': self.par['bus']}
    
    def I_inj(self, x, v):
        return (self.I_d(x, v) - 1j*self.I_q(x, v))*np.exp(1j*self.pll.output(x, v))
    
    def state_list(self): 
        """
        All states in pu.(?)

        i_d: d-axis current, first-order approximation of EM dynamics
        i_q: q-axis current -------------""--------------
        x_p: p-control integral
        x_v: v-control integral
        x_pll: pll q-axis integral
        angle: pll angle
        """
        return ['i_d', 'i_q', 'x_p', 'x_v', 'x_pll', 'angle']

    def input_list(self):
        '''
        P_setp: active power setpoint
        V_setp: Voltage setpoint
        '''
        return ['P_setp', 'V_setp']
    
    def init_from_load_flow(self, x_0, v_0, S):
        X = self.local_view(x_0)

        self._input_values['P_setp'] = self.par['P_setp']/self.par['S_n'] #p.u
        self._input_values['V_setp'] = self.par['V_setp']

        v0 = v_0[self.bus_idx_red['terminal']] #tror dette er i p.u
        s0 = S[self.bus_idx_red['terminal']] #blir dette i MW eller p.u.

        X['x_p'] = s0.real/(abs(v0))
        X['x_v'] = s0.imag/(abs(v0))
        X['i_d'] = X['x_p']
        X['i_q'] = -X['x_v']
        X['x_pll'] = 0
        X['angle'] = np.angle(v0)
        X['rocof'] = 0

    

    def state_derivatives(self, dx, x, v):
        dX = self.local_view(dx)
        X = self.local_view(x)
        par = self.par



        dp = self.P_setp(x,v) - self.p_e(x,v)
        dv = self.V_setp(x,v) - abs(self.v_t(x,v))
        i_d_ref = dp*par['k_p'] + X['x_p']
        i_q_ref = -dv*par['k_v'] - X['x_v']

        #limiters
        i_ref = (i_d_ref+1j*i_q_ref)
        i_ref = i_ref*par['i_max']/np.maximum(par['i_max'],abs(i_ref))
        X['x_p'] = np.maximum(np.minimum(X['x_p'],par['i_max']),-par['i_max'])
        X['x_v'] = np.maximum(np.minimum(X['x_v'],par['i_max']),-par['i_max'])

        #differential equations
        dX['i_d'][:] = 1 / (par['T_i']) * (i_ref.real - X['i_d'])
        dX['i_q'][:] = 1 / (par['T_i']) * (i_ref.imag - X['i_q'])
        dX['x_p'][:] = par['k_p']/par['T_p'] * dp
        dX['x_v'][:] = par['k_v']/par['T_v'] * dv
        dX['x_pll'][:] = par['k_pll']/par['T_pll'] * (self.v_q(x,v))
        dX['angle'][:] = X['x_pll']+par['k_pll']*self.v_q(x,v)



        return None 
    
    def current_injections(self, x, v):
        i_n_r = self.par['S_n'] / self.sys_par['s_n']
        return self.bus_idx_red['terminal'], self.i_inj(x, v) * i_n_r
    
    def i_inj(self, x, v):
        X = self.local_view(x)
        return (X['i_d'] + 1j * X['i_q']) * np.exp(1j*X['angle'])
    
    def v_t(self, x, v):
        return v[self.bus_idx_red['terminal']]

    def s_e(self, x, v):
        # Apparent power in p.u. (generator base units)
        return self.v_t(x, v)*np.conj(self.i_inj(x, v))

    def v_q(self,x,v):
        return (self.v_t(x,v)*np.exp(-1j*self.local_view(x)['angle'])).imag
    
    def p_e(self, x, v):
        return self.s_e(x,v).real

    def q_e(self, x, v):
        return self.s_e(x,v).imag
    


class VSC_PQ_SI(DAEModel):
    """
    Instantiate:
    'vsc': {
            'VSC_PQ': [
                ['name', 'bus', 'S_n', 'p_ref', 'q_ref',  'k_p', 'k_q', 'T_p', 'T_q', 'k_pll','T_pll', 'T_i', 'i_max', 'K_SI, 'T_rocof'],
                ['VSC1', 'B1',    50,     1,       0,       1,      1,    0.1,   0.1,     5,      1,      0.01,    1.2, 10, 1],
            ],
        }
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.bus_idx = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])
        self.bus_idx_red = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])

    # region Definitions

    def load_flow_pq(self):
        return self.bus_idx['terminal'], -self.par['p_ref']*self.par['S_n'], -self.par['q_ref']*self.par['S_n']

    def int_par_list(self):
        return ['f']

    def bus_ref_spec(self):
        return {'terminal': self.par['bus']}

    # endregion

    def state_list(self):
        """
        All states in pu.

        i_d: d-axis current, first-order approximation of EM dynamics
        i_q: q-axis current -------------""--------------
        x_p: p-control integral
        x_q: q-control integral
        x_pll: pll q-axis integral
        angle: pll angle

        rocof: df/dt (from PLL)
        """
        return ['i_d', 'i_q', 'x_p', 'x_q', 'x_pll', 'angle', 'rocof']

    def input_list(self):
        """
        All values in pu.

        p_ref: outer loop active power setpoint
        q_ref: outer loop reactive power setpoint
        """
        return ['p_ref', 'q_ref']

    def state_derivatives(self, dx, x, v):
        dX = self.local_view(dx)
        X = self.local_view(x)
        par = self.par

        dp = self.p_ref(x,v) - self.p_e(x, v) - par['K_SI']*X['rocof']
        dq = -self.q_ref(x,v) + self.q_e(x, v)
        i_d_ref = dp * par['k_p'] + X['x_p']
        i_q_ref = dq * par['k_q'] + X['x_q']

        # Limiters
        i_ref = (i_d_ref+1j*i_q_ref)
        i_ref = i_ref*par['i_max']/np.maximum(par['i_max'],abs(i_ref))
        X['x_p'] = np.maximum(np.minimum(X['x_p'],par['i_max']),-par['i_max'])
        X['x_q'] = np.maximum(np.minimum(X['x_q'],par['i_max']),-par['i_max'])

        dX['i_d'][:] = 1 / (par['T_i']) * (i_ref.real - X['i_d'])
        dX['i_q'][:] = 1 / (par['T_i']) * (i_ref.imag - X['i_q'])
        dX['x_p'][:] = par['k_p'] / (par['T_p']) * dp
        dX['x_q'][:] = par['k_q'] / (par['T_q']) * dq
        dX['x_pll'][:] = par['k_pll'] / (par['T_pll']) * (self.v_q(x,v))
        dX['angle'][:] = X['x_pll']+par['k_pll']*self.v_q(x,v)

        rocof = par['k_pll'] / (par['T_pll']) * (self.v_q(x,v))
        dX['rocof'][:] = 1/par['T_rocof']*(rocof-X['rocof'])
        return

    def init_from_load_flow(self, x_0, v_0, S):
        X = self.local_view(x_0)

        self._input_values['p_ref'] = self.par['p_ref']#dette kan være feil OBSOBS
        self._input_values['q_ref'] = self.par['q_ref']

        v0 = v_0[self.bus_idx_red['terminal']]

        X['i_d'] = self.par['p_ref']/abs(v0)
        X['i_q'] = self.par['q_ref']/abs(v0)
        X['x_p'] = X['i_d']
        X['x_q'] = X['i_q']
        X['x_pll'] = 0
        X['angle'] = np.angle(v0)

        X['rocof'] = 0 #denne er jeg litt usikker på 


    def current_injections(self, x, v):
        i_n_r = self.par['S_n'] / self.sys_par['s_n']
        return self.bus_idx_red['terminal'], self.i_inj(x, v) * i_n_r

    # region Utility methods
    def i_inj(self, x, v):
        X = self.local_view(x)
        #v_t = self.v_t(x,v)
        return (X['i_d'] + 1j * X['i_q']) * np.exp(1j*X['angle'])

    def v_t(self, x, v):
        return v[self.bus_idx_red['terminal']]

    def s_e(self, x, v):
        # Apparent power in p.u. (generator base units)
        return self.v_t(x, v)*np.conj(self.i_inj(x, v))

    def v_q(self,x,v):
        return (self.v_t(x,v)*np.exp(-1j*self.local_view(x)['angle'])).imag
    def p_e(self, x, v):
        return self.s_e(x,v).real

    def q_e(self, x, v):
        return self.s_e(x,v).imag

    # endregion


class VSC_PQ(DAEModel):
    """
    Instantiate:
    'vsc': {
            'VSC_PQ': [
                ['name', 'bus', 'S_n', 'p_ref', 'q_ref',  'k_p', 'k_q', 'T_p', 'T_q', 'k_pll','T_pll', 'T_i', 'i_max'],
                ['VSC1', 'B1',    50,     1,       0,       1,      1,    0.1,   0.1,     5,      1,      0.01,    1.2],
            ],
        }
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.bus_idx = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])
        self.bus_idx_red = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])

    # region Definitions

    def load_flow_pq(self):
        return self.bus_idx['terminal'], -self.par['p_ref']*self.par['S_n'], -self.par['q_ref']*self.par['S_n']

    def int_par_list(self):
        return ['f']

    def bus_ref_spec(self):
        return {'terminal': self.par['bus']}

    # endregion

    def state_list(self):
        """
        All states in pu.

        i_d: d-axis current, first-order approximation of EM dynamics
        i_q: q-axis current -------------""--------------
        x_p: p-control integral
        x_q: q-control integral
        x_pll: pll q-axis integral
        angle: pll angle
        """
        return ['i_d', 'i_q', 'x_p', 'x_q', 'x_pll', 'angle']

    def input_list(self):
        """
        All values in pu.

        p_ref: outer loop active power setpoint
        q_ref: outer loop reactive power setpoint
        """
        return ['p_ref', 'q_ref']

    def state_derivatives(self, dx, x, v):
        dX = self.local_view(dx)
        X = self.local_view(x)
        par = self.par

        dp = self.p_ref(x,v) - self.p_e(x, v)
        dq = -self.q_ref(x,v) + self.q_e(x, v)
        i_d_ref = dp * par['k_p'] + X['x_p']
        i_q_ref = dq * par['k_q'] + X['x_q']

        # Limiters
        i_ref = (i_d_ref+1j*i_q_ref)
        i_ref = i_ref*par['i_max']/np.maximum(par['i_max'],abs(i_ref))
        X['x_p'] = np.maximum(np.minimum(X['x_p'],par['i_max']),-par['i_max'])
        X['x_q'] = np.maximum(np.minimum(X['x_q'],par['i_max']),-par['i_max'])

        dX['i_d'][:] = 1 / (par['T_i']) * (i_ref.real - X['i_d'])
        dX['i_q'][:] = 1 / (par['T_i']) * (i_ref.imag - X['i_q'])
        dX['x_p'][:] = par['k_p'] / (par['T_p']) * dp
        dX['x_q'][:] = par['k_q'] / (par['T_q']) * dq
        dX['x_pll'][:] = par['k_pll'] / (par['T_pll']) * (self.v_q(x,v))
        dX['angle'][:] = X['x_pll']+par['k_pll']*self.v_q(x,v)
        #dX['angle'][:] = 0
        return

    def init_from_load_flow(self, x_0, v_0, S):
        X = self.local_view(x_0)

        self._input_values['p_ref'] = self.par['p_ref']
        self._input_values['q_ref'] = self.par['q_ref']

        v0 = v_0[self.bus_idx_red['terminal']]

        X['i_d'] = self.par['p_ref']/abs(v0)
        X['i_q'] = self.par['q_ref']/abs(v0)
        X['x_p'] = X['i_d']
        X['x_q'] = X['i_q']
        X['x_pll'] = 0
        X['angle'] = np.angle(v0)

    def current_injections(self, x, v):
        i_n_r = self.par['S_n'] / self.sys_par['s_n']
        return self.bus_idx_red['terminal'], self.i_inj(x, v) * i_n_r

    # region Utility methods
    def i_inj(self, x, v):
        X = self.local_view(x)
        #v_t = self.v_t(x,v)
        return (X['i_d'] + 1j * X['i_q']) * np.exp(1j*X['angle'])

    def v_t(self, x, v):
        return v[self.bus_idx_red['terminal']]

    def s_e(self, x, v):
        # Apparent power in p.u. (generator base units)
        return self.v_t(x, v)*np.conj(self.i_inj(x, v))

    def v_q(self,x,v):
        return (self.v_t(x,v)*np.exp(-1j*self.local_view(x)['angle'])).imag
    def p_e(self, x, v):
        return self.s_e(x,v).real

    def q_e(self, x, v):
        return self.s_e(x,v).imag
