from dynpssimpy.solvers import EulerDAE

class EulerDAE_SDE(EulerDAE):
    def __init__(self, *args, **kwargs):
        '''
        First implementation of a Stochastic Differential Equation Solver, based on following code snippet:
        https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method
        '''
        super(*args, **kwargs)
        

    def step(self):
        if self.t < self.t_end:
            self.x[:] = self.x + self.f(self.t, self.x, self.v)*self.dt + self.b(self.t, self.x, self.v)
            self.t += self.dt
            self.v[:] = self.g_inv(self.t, self.x)

        else:
            print('End of simulation time reached.')