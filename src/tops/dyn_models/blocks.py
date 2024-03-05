from tops.dyn_models.utils import DAEModel, output
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
    '''
            ______________	
           |               |
           |       1       |
    u ---->|  -----------  |----> y
           |    1 + s*T    |
           |_______________|


    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.zero_idx = self.par['T']==0
    
    def state_list(self):
        return ['x']

    @output
    def output(self, x, v):
        X = self.local_view(x)
        out = X['x']
        if np.any(self.zero_idx):
            out[self.zero_idx] = self.input(x, v)[self.zero_idx]
        return out

    def initialize(self, x0, v0, output_value):
        X0 = self.local_view(x0)
        X0['x'][:] = output_value
        return output_value

    def state_derivatives(self, dx, x, v):
        dX = self.local_view(dx)
        X = self.local_view(x)
        
        coeff = ~self.zero_idx/(self.par['T']+self.zero_idx)  # 1/T if T is not zero, 0 otherw
        dX['x'][:] = coeff*(self.input(x, v) - X['x'])


class TimeConstantVar(TimeConstant):
    '''
            ______________	
           |               |
           |       1       |
    u ---->|  -----------  |----> y
           |    K + s*T    |
           |_______________|


    '''
    def initialize(self, x0, v0, output_value):
        X0 = self.local_view(x0)
        X0['x'][:] = output_value
        return self.par['K']*output_value

    def state_derivatives(self, dx, x, v):
        dX = self.local_view(dx)
        X = self.local_view(x)
        # Check if T=0?
        dX['x'][:] = 1/self.par['T']*(self.input(x, v) - self.par['K']*X['x'])


class TimeConstantGain(TimeConstant):
    def output(self, x, v):
        return self.par['K']*super().output(x, v)
    
    def initialize(self, x0, v0, output_value):
        return super().initialize(x0, v0, output_value/self.par['K'])


class TimeConstantLims(DAEModel):
    '''
	                 ___ V_max
            ________/_____	
           |               |
           |       1       |
    u ---->|  -----------  |----> y
           |   1 + s*T_2   |
           |_______________|
               ___/
          V_min

    '''
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

class TimeConstantGainLims(TimeConstantLims):
    '''
	                 ___ V_max
            ________/_____	
           |               |
           |       K       |
    u ---->|  -----------  |----> y
           |   1 + s*T_2   |
           |_______________|
               ___/
          V_min

    '''
    def state_derivatives(self, dx, x, v):
        dX = self.local_view(dx)
        X = self.local_view(x)

        dX['x'][:] = 1/self.par['T']*(self.par['K']*self.input(x, v) - X['x'])

        # Lims on state variable (clamping)
        lower_lim_idx = (X['x'] <= self.par['V_min']) & (dX['x'] < 0)
        dX['x'][lower_lim_idx] *= 0

        upper_lim_idx = (X['x'] >= self.par['V_max']) & (dX['x'] > 0)
        dX['x'][upper_lim_idx] *= 0
    
    def initialize(self, x0, v0, output_value):
        return super().initialize(x0, v0, output_value/self.par['K'])

class LeadLag(DAEModel):
    '''
            _______________
           |               |
           |   1 + s*T_1   |
    u ---->|  -----------  |----> y
           |   1 + s*T_2   |
           |_______________|
     
    '''
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
        return (self.par['T_1']/self.par['T_2']*self.input(x, v) + X['x'])

    def state_derivatives(self, dx, x, v):
        dX = self.local_view(dx)
        X = self.local_view(x)
        dX['x'][:] = self.input(x, v)/self.par['T_2']

    def initialize(self, x0, v0, output_value):
        X = self.local_view(x0)
        X['x'] = output_value
        return np.zeros(self.n_units)
    

class PIRegulator2Lims(PIRegulator2):
    @output
    def output(self, x, v):
        output_before_limiter = super().output(x, v)
        return np.minimum(np.maximum(output_before_limiter, self.par['x_min']), self.par['x_max'])

    def state_derivatives(self, dx, x, v):
        super().state_derivatives(dx, x, v)

        # Lims on state variable (clamping)
        dX = self.local_view(dx)
        X = self.local_view(x)
        lower_lim_idx = (X['x'] <= self.par['x_min']) & (dX['x'] < 0)
        dX['x'][lower_lim_idx] *= 0

        upper_lim_idx = (X['x'] >= self.par['x_max']) & (dX['x'] > 0)
        dX['x'][upper_lim_idx] *= 0

    def initialize(self, x0, v0, output_value):
        super().initialize(x0, v0, output_value)
        X0 = self.local_view(x0)
        X0['x'][:] = np.minimum(np.maximum(X0['x'], self.par['x_min']), self.par['x_max'])
        return np.zeros(self.n_units)   


class WashoutGain(DAEModel):
    '''
            _______________
           |               |
           |      s*K      |
    u ---->|  -----------  |----> y
           |    1 + s*T    |
           |_______________|
     
    sK/(1+sT) = y/u
    u*sK = y + sTy
    s(Ku - Ty) = y
    x = Ku - Ty
    sx = y
    y = 1/T(Ku - x)
    '''
    def state_list(self):
        return ['x']

    @output
    def output(self, x, v):
        X = self.local_view(x)
        return 1/self.par['T_w']*(self.par['K']*self.input(x, v) - X['x'])

    def state_derivatives(self, dx, x, v):
        dX = self.local_view(dx)
        dX['x'][:] = self.output(x, v)

    # def initialize(self, x0, v0, output_value):
        # dx = 0 => x = Ku
        # pass


class Saturation(DAEModel):
    '''
    Not verified!!
    '''
    @output
    def output(self, x, v):
        U = self.input(x, v)
        
        E1 = self.par['E_1']
        SE1 = self.par['S_e1']
        E2 = self.par['E_2']
        SE2 = self.par['S_e2']

        sqrt = np.sqrt
        K = SE1/SE2
        A = sqrt(E1*E2)*(sqrt(E1) - sqrt(E2*K))/(sqrt(E2) - sqrt(E1*K))
        B = SE2*(sqrt(E2) - sqrt(E1*K))**2/(E1 - E2)**2

        
        with np.errstate(divide='ignore'):
            SE = B*(U - A)**2/U
        SE[U <= 0] = 0
        
        return SE
    

class Backlash(DAEModel):
    """Based on PowerFactory implementation"""
    def state_list(self):
        return ['x']
    
    def state_derivatives(self, dx, x, v):
        dX = self.local_view(dx)
        X = self.local_view(x)
        db = self.par['db']
        if db <= 0:
            return db*0

        y = self.input(x, v)
        x = X['x']

        if (y - x) >= db:
            d = (y - x) - db
        elif (y - x) <= -db:
            d = (y - x) + db
        else:
            d = 0

        dX['x'][:] = d/0.01

    @output
    def output(self, x, v):
        X = self.local_view(x)
        return X['x']
    
    def initialize(self, x0, v0, output_value):
        X0 = self.local_view(x0)
        X0['x'][:] = output_value
        return output_value