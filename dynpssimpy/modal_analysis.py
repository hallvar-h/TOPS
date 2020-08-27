import numpy as np
import dynpssimpy.dynamic as dps
import dynpssimpy.plotting as dps_plt
import dynpssimpy.utility_functions as utils
import matplotlib.pyplot as plt


class PowerSystemModelLinearization:
    def __init__(self, eq):
        self.eq = eq
        self.eps = 1e-10
        pass

    def linearize(self, eq=None, x0=np.array([])):
        if eq:
            self.eq = eq

        self.x0 = x0 if len(x0) > 0 else self.eq.x0
        self.a = utils.jacobian_num(lambda x: self.eq.ode_fun(0, x), self.x0, eps=self.eps)
        self.n = self.a.shape[0]
        self.eigs, evs = np.linalg.eig(self.a)
        # self.lev = np.conj(evs).T
        self.rev = evs
        self.lev = np.linalg.inv(self.rev)
        self.damping = -self.eigs.real / abs(self.eigs)
        self.freq = self.eigs.imag / (2 * np.pi)

    def linearize_inputs(self, inputs):
        eq = self.eq
        b = np.zeros((len(eq.x0), 0))
        for inp in inputs:
            var = getattr(eq, inp[0])
            index = inp[1]
            if not index:
                index = range(len(var))

            if len(inp) == 3:
                if inp[2] == 'Complex' or inp[2] == 'imag':
                    mod = 1j
            else:
                mod = 1

            for i in index:
                var_0 = var[i]
                var[i] = var_0 + self.eps*mod
                f_1 = eq.ode_fun(0, eq.x0)
                var[i] = var_0 - self.eps*mod
                f_2 = eq.ode_fun(0, eq.x0)
                var[i] = var_0
                b = np.hstack([b, ((f_1 - f_2) / (2 * self.eps))[:, None]])
        # self.b = b
        return b

    def linearize_inputs_v2(self, input_desc):
        eq = self.eq
        eps = self.eps
        b = np.zeros((len(eq.x0), len(input_desc)))
        for i, inp_ in enumerate(input_desc):
            b_tmp = np.zeros(len(eq.x0))
            for inp__ in inp_:
                var = getattr(eq, inp__[0])
                index = inp__[1]
                gain = inp__[2] if len(inp__) == 3 else 1

                var_0 = var[index]
                var[index] = var_0 + eps * gain
                f_1 = eq.ode_fun(0, eq.x0)
                var[index] = var_0 - eps * gain
                f_2 = eq.ode_fun(0, eq.x0)
                var[index] = var_0
                b_tmp += ((f_1 - f_2) / (2 * eps))
            b[:, i] = b_tmp  # np.hstack([b, ((f_1 - f_2) / (2 * eps))[:, None]])
        # self.b = b
        return b

    def get_mode_idx(self, mode_type=['em', 'non_conj'], damp_threshold=1, sorted=True):
        eigs = self.eigs
        idx = np.ones(len(eigs), dtype=bool)
        print('h')
        if not isinstance(mode_type, list):
            mode_type = [mode_type]

        for mt in mode_type:
            if mt == 'em':
                idx *= (abs(eigs.imag) / (2 * np.pi) > 0.1) & (abs(eigs.imag) / (2 * np.pi) < 3)
            if mt == 'non_conj':
                idx *= eigs.imag >= 0

        idx *= self.damping < damp_threshold

        idx = np.where(idx)[0]
        if sorted:
            idx = idx[np.argsort(self.damping[idx])]
        return idx


if __name__ == '__main__':

    import ps_models.k2a as model_data
    import importlib
    importlib.reload(dps)

    ps = dps.PowerSystemModel(model_data.load())
    ps.power_flow()
    ps.init_dyn_sim()

    # Perform system linearization
    ps_lin = PowerSystemModelLinearization(ps)
    ps_lin.linearize()

    # Plot eigenvalues
    dps_plt.plot_eigs(ps_lin.eigs)

    # Get mode shape for electromechanical modes
    mode_idx = ps_lin.get_mode_idx(['em'], damp_threshold=0.3)
    rev = ps_lin.rev
    mode_shape = rev[np.ix_(ps.speed_idx, mode_idx)]

    # Plot mode shape
    fig, ax = plt.subplots(1, mode_shape.shape[1], subplot_kw={'projection': 'polar'})
    for ax_, ms in zip(ax, mode_shape.T):
        dps_plt.plot_mode_shape(ms, ax=ax_, normalize=True)
