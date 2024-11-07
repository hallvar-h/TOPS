from tops.dyn_models.blocks import *

class PLL1(DAEModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.bus_idx = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])
        self.bus_idx_red = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])

    def bus_ref_spec(self):
        return {'terminal': self.par['bus']}

    def add_blocks(self):
        p = self.par
        self.filter = TimeConstant(T=p['T_filter'])

        def angle_measurement(x, v):
            v_angle = np.angle(v[self.bus_idx_red['terminal']])

            # np.unwrap(np.vstack([np.array([0, 3 * np.pi, 0]), np.array([0, np.pi + 1e-10 + 2 * np.pi, 0])]), axis=0)[1, :]
            return np.unwrap(np.vstack([self.output(x, v), v_angle]), axis=0)[1, :]

        # self.filter.input = lambda x, v: np.angle(v[self.bus_idx_red['terminal']])
        self.filter.input = angle_measurement
        self.output = lambda x, v: self.filter.output(x, v)

    # @output
    def freq_est(self, x, v):
        X = self.filter.local_view(x)
        dX = 1/self.filter.par['T']*(self.filter.input(x, v) - X['x'])
        return dX/(2 * np.pi * self.sys_par['f_n'])

    def init_from_load_flow(self, x_0, v_0, S):
        output_value = np.angle(v_0[self.bus_idx_red['terminal']])
        self.filter.initialize(x_0, v_0, output_value)


class PLL2(PLL1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.bus_idx = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])
        self.bus_idx_red = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])

    def bus_ref_spec(self):
        return {'terminal': self.par['bus']}
    
    def add_blocks(self):
        p = self.par
        self.pi = PIRegulator(K_p=p['K_p'], K_i=p['K_i'])
        self.integrator = Integrator(n_units=self.n_units)

        self.v_measured = lambda x, v: v[self.bus_idx_red['terminal']]
        self.phi = self.integrator.output
        self.pi.input = lambda x, v: -self.v_measured(x, v).real*np.sin(self.phi(x, v)) + self.v_measured(x, v).imag*np.cos(self.phi(x, v))
        self.integrator.input = self.pi.output

        self.freq_est = lambda x, v: self.integrator.input(x, v)/(2 * np.pi * self.sys_par['f_n'])

        self.output = self.phi

    def init_from_load_flow(self, x_0, v_0, S):
        output_value = np.angle(v_0[self.bus_idx_red['terminal']])
        # self.pi.initialize(
        x_0, v_0, self.integrator.initialize(x_0, v_0, output_value)
        # 

class PLL1_nora(DAEModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.bus_idx = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])
        self.bus_idx_red = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])

    def bus_ref_spec(self):
        return {'terminal': self.par['bus']}

    def add_blocks(self):
        p = self.par
        self.filter = TimeConstant(T=p['T_filter'])

        def angle_measurement(x, v):
            v_angle = np.angle(v[self.bus_idx_red['terminal']])

            # np.unwrap(np.vstack([np.array([0, 3 * np.pi, 0]), np.array([0, np.pi + 1e-10 + 2 * np.pi, 0])]), axis=0)[1, :]
            return np.unwrap(np.vstack([self.output(x, v), v_angle]), axis=0)[1, :]

        # self.filter.input = lambda x, v: np.angle(v[self.bus_idx_red['terminal']])
        self.filter.input = angle_measurement
        self.output = lambda x, v: self.filter.output(x, v)

    # @output
    def freq_est(self, x, v):
        X = self.filter.local_view(x)
        dX = 1/self.filter.par['T']*(self.filter.input(x, v) - X['x'])
        return dX/(2 * np.pi * 50)

    def init_from_load_flow(self, x_0, v_0, S):
        output_value = np.angle(v_0[self.bus_idx_red['terminal']])
        self.filter.initialize(x_0, v_0, output_value)
