from tops.solvers import EulerDAE
import numpy as np

class EulerDAE_SDE(EulerDAE):
    def __init__(self, *args, dim_w=0, **kwargs):
        '''
        First implementation of a Stochastic Differential Equation Solver, based on following code snippet:
        https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method
        '''
        super().__init__(*args, **kwargs)
        self.dim_w = dim_w
        self.b = lambda t, x, v: np.zeros((len(self.x), self.dim_w))
        self.dw = np.zeros(self.dim_w)
        

    def step(self):
        self.dw = np.random.normal(loc=0.0, scale=np.sqrt(self.dt), size=self.dim_w)
        if self.t < self.t_end:
            self.x[:] = self.x + self.f(self.t, self.x, self.v)*self.dt + np.dot(self.b(self.t, self.x, self.v), self.dw)
            self.t += self.dt
            self.v[:] = self.g_inv(self.t, self.x)

        else:
            print('End of simulation time reached.')