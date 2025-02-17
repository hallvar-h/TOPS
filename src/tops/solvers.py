import numpy as np


class Euler:
    def __init__(self, f, t0, x0, t_end=np.inf, dt=5e-3, **kwargs):
        self.f = f
        self.t = t0
        self.x = x0.copy()
        self.y = self.x
        self.t_end = t_end
        self.dt = dt

        for key, value in kwargs.items():
            if key == 'max_step':
                self.dt = value

    def step(self):
        if self.t < self.t_end:
            self.x[:] = self.x + self.f(self.t, self.x)*self.dt
            self.t += self.dt


class EulerDAE(Euler):
    def __init__(self, f, g_inv, *args, **kwargs):
        '''
        Similar to Euler solver-class, but ensures that algebraic equations (stored in self.v) are always updated at the end of each time step.
        :param f: Function that takes time, states and algebraic variables (t, x and v) as arguments and returns state
        derivatives.
        :param g_inv: Function that takes time and states as arguments and solves algebraic equations (i.e. returns
        bus voltages of buses in reduced system).
        :param args:
        :param kwargs:
        '''
        super().__init__(f, *args, **kwargs)
        self.g_inv = g_inv
        self.v = self.g_inv(self.t, self.x)

    def step(self):
        if self.t < self.t_end:
            self.x[:] = self.x + self.f(self.t, self.x, self.v) * self.dt
            self.t += self.dt
            self.v[:] = self.g_inv(self.t, self.x, self.v)

        else:
            print('End of simulation time reached.')


class ModifiedEuler(Euler):
    def __init__(self, *args, n_it=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_it = n_it

    def step(self):
        if self.t < self.t_end:
            dxdt_0 = self.f(self.t, self.x)
            x_1 = self.x + dxdt_0*self.dt
            for _ in range(self.n_it):
                dxdt_1 = self.f(self.t + self.dt, x_1)
                dxdt_est = (dxdt_0 + dxdt_1) / 2
                x_1 = self.x + dxdt_est*self.dt

            self.x[:] = x_1
            self.t += self.dt

        else:
            print('End of simulation time reached.')


class ModifiedEulerDAE(EulerDAE):
    def __init__(self, *args, n_it=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_it = n_it
        self.f_ode = lambda t, x: self.f(t, x, self.g_inv(t, x))

    def step(self):
        if self.t < self.t_end:
            dxdt_0 = self.f(self.t, self.x, self.v)
            x_1 = self.x + dxdt_0*self.dt
            for _ in range(self.n_it):
                dxdt_1 = self.f(self.t + self.dt, x_1, self.v)
                dxdt_est = (dxdt_0 + dxdt_1) / 2
                x_1 = self.x + dxdt_est*self.dt

            self.x[:] = x_1
            self.v[:] = self.g_inv(self.t, self.x, self.v)
            self.t += self.dt

        else:
            print('End of simulation time reached.')


class SimpleRK4(Euler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self):
        x = self.x
        t = self.t
        dt = self.dt

        if t < self.t_end:
            k_1 = self.f(t, x)
            k_2 = self.f(t + dt / 2, x + (dt / 2) * k_1)
            k_3 = self.f(t + dt / 2, x + (dt / 2) * k_2)
            k_4 = self.f(t + dt, x + dt * k_3)

            self.x[:] = x + (dt / 6) * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
            self.t = t + dt
        else:
            print('End of simulation time reached.')


