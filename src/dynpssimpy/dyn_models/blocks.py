from dynpssimpy.dyn_models.utils import DAEModel, output
import numpy as np


class PIRegulator(DAEModel):

    def add_blocks(self):
        self.integrator = Integrator(n_units=self.n_units)
        self.integrator.input = lambda x, v: self.input(x, v)

    @output
    def output(self, x, v):
        # X = self.local_view(x)
        return self.par['K_p']*self.input(x, v) + self.par['K_i']*self.integrator.output(x, v)

    def initialize(self, x0, v0, output_value):
        input_value = self.integrator.initialize(x0, v0, output_value/self.par['K_i'])
        return input_value


class Integrator(DAEModel):
    def state_list(self):
        return ['x_i']

    @output
    def output(self, x, v):
        X = self.local_view(x)
        return X['x_i']

    def state_derivatives(self, dx, x, v):
        dX = self.local_view(dx)
        X = self.local_view(x)

        dX['x_i'][:] = self.input(x, v)

    def initialize(self, x0, v0, output_value):
        X0 = self.local_view(x0)
        X0['x_i'][:] = output_value
        input_value = output_value
        return input_value*0

    
class Integrator2(Integrator):
    """
    Same as Integrator, but with time constant.
    """
    @output
    def output(self, x, v):
        X = self.local_view(x)
        return X['x_i']/self.par['T']

    def initialize(self, x0, v0, output_value):
        X0 = self.local_view(x0)
        X0['x_i'][:] = output_value*self.par['T']
        input_value = output_value*self.par['T']
        return input_value*0


class Gain(DAEModel):
    @output
    def output(self, x, v):
        # X = self.local_view(x)
        return self.par['K']*self.input(x, v)

    def initialize(self, x0, v0,output_value):
        return output_value/self.par['K']


class Limiter(DAEModel):
    @output
    def output(self, x, v):
        # X = self.local_view(x)
        return np.minimum(np.maximum(self.input(x, v), self.par['Min']), self.par['Max'])


class Washout(DAEModel):
    def state_list(self):
        return ['x']

    @output
    def output(self, x, v):
        X = self.local_view(x)
        return (self.input(x, v) - X['x'])/self.par['T_w']

    def state_derivatives(self, dx, x, v):
        dX = self.local_view(dx)
        X = self.local_view(x)

        dX['x'][:] = self.output(x, v)


class TimeConstant(DAEModel):
    def state_list(self):
        return ['x']

    @output
    def output(self, x, v):
        X = self.local_view(x)
        return X['x']

    def initialize(self, x0, v0,output_value):
        X0 = self.local_view(x0)
        X0['x'][:] = output_value
        return output_value

    def state_derivatives(self, dx, x, v):
        dX = self.local_view(dx)
        X = self.local_view(x)

        dX['x'][:] = 1/self.par['T']*(self.input(x, v) - X['x'])


class TimeConstantLims(DAEModel):
    def state_list(self):
        return ['x']

    @output
    def output(self, x, v):
        X = self.local_view(x)
        return np.minimum(np.maximum(X['x'], self.par['V_min']), self.par['V_max'])

    def initialize(self, x0, v0,output_value):
        X0 = self.local_view(x0)
        X0['x'][:] = np.minimum(np.maximum(output_value, self.par['V_min']), self.par['V_max'])
        return output_value

    def state_derivatives(self, dx, x, v):
        dX = self.local_view(dx)
        X = self.local_view(x)

        dX['x'][:] = 1/self.par['T']*(self.input(x, v) - X['x'])

        # Lims on state variable (clamping)
        lower_lim_idx = (X['x'] <= self.par['V_min']) & (dX['x'] < 0)
        dX['x'][lower_lim_idx] *= 0

        upper_lim_idx = (X['x'] >= self.par['V_max']) & (dX['x'] > 0)
        dX['x'][upper_lim_idx] *= 0


class LeadLag(DAEModel):
    def state_list(self):
        return ['x']

    @output
    def output(self, x, v):
        X = self.local_view(x)
        return 1/self.par['T_2']*(self.par['T_1']*self.input(x, v) - X['x'])

    def initialize(self, x0, v0,output_value):
        X0 = self.local_view(x0)
        p = self.par

        X0['x'][:] = p['T_1']*output_value - p['T_2']*output_value
        return output_value

    def state_derivatives(self, dx, x, v):
        dX = self.local_view(dx)
        X = self.local_view(x)

        dX['x'][:] = (self.par['T_1']/self.par['T_2'] - 1)*self.input(x, v) - 1/self.par['T_2']*X['x']


class PIRegulator2(DAEModel):
    def state_list(self):
        return ['x']

    @output
    def output(self, x, v):
        X = self.local_view(x)
        return 1/self.par['T_2']*(self.par['T_1']*self.input(x, v) + X['x'])

    def state_derivatives(self, dx, x, v):
        dX = self.local_view(dx)
        X = self.local_view(x)
        dX['x'][:] = self.input(x, v)

    def initialize(self, x0, v0, output_value):
        X = self.local_view(x0)
        X['x'] = self.par['T_2']/self.par['T_1']*output_value
        return np.zeros(self.n_units)