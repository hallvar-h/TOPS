import numpy as np
import dynpssimpy.plotting as dps_plt
import dynpssimpy.utility_functions as utils
import matplotlib.pyplot as plt


class PowerSystemModelLinearization:
    def __init__(self, ps):
        self.ps = ps
        self.eps = 1e-10
        self.linearize_inputs_v2 = self.linearize_inputs

    def linearize(self, ps=None, x0=np.array([]), input_description=np.array([]), output_description=np.array([])):
        # Linearizes non-linear ODEs at operating point x0.
        if ps:
            self.ps = ps

        self.x0 = x0 if len(x0) > 0 else self.ps.x0
        self.a = utils.jacobian_num(lambda x: self.ps.ode_fun(0, x), self.x0, eps=self.eps)
        self.n = self.a.shape[0]
        self.eigs, evs = np.linalg.eig(self.a)

        # Right/left rigenvectors (rev/lev)
        self.rev = evs
        self.lev = np.linalg.inv(self.rev)
        self.damping = -self.eigs.real / abs(self.eigs)
        self.freq = self.eigs.imag / (2 * np.pi)

        if len(input_description) > 0:
            self.b = self.linearize_inputs(input_description)

        if len(input_description) > 0:
            self.c = self.linearize_outputs(output_description)

    def linearize_inputs(self, input_description):
        # Perturbs values in PowerSystemModel-object, as indicated by "input_description", and computes
        # the input matrix (or vector) "b" from the change in states.
        ps = self.ps
        eps = self.eps
        b = np.zeros((len(ps.x0), len(input_description)))
        for i, inp_ in enumerate(input_description):
            b_tmp = np.zeros(len(ps.x0))
            for inp__ in inp_:
                var = getattr(ps, inp__[0])
                index = inp__[1]
                gain = inp__[2] if len(inp__) == 3 else 1

                var_0 = var[index]
                var[index] = var_0 + eps * gain
                f_1 = ps.ode_fun(0, ps.x0)
                var[index] = var_0 - eps * gain
                f_2 = ps.ode_fun(0, ps.x0)
                var[index] = var_0
                b_tmp += ((f_1 - f_2) / (2 * eps))
            b[:, i] = b_tmp
        return b

    def linearize_outputs(self, output_description):
        # Perturbs states in PowerSystemModel-object, as indicated by "output_description", and computes
        # the output matrix (or vector) "c" from the change in output.
        ps = self.ps
        eps = self.eps
        x = ps.x0.copy()
        c = np.zeros((len(output_description), len(ps.x0)), dtype=complex)
        for i, outp_ in enumerate(output_description):
            c_tmp = np.zeros(ps.n_states, dtype=complex)
            for j in range(ps.n_states):
                for outp__ in outp_:
                    var = outp__[0]
                    index = outp__[1]
                    gain = outp__[2] if len(outp__) == 3 else 1

                    x_1 = x.copy()
                    x_2 = x.copy()

                    x_1[j] += eps
                    x_2[j] -= eps

                    ps.ode_fun(0, x_1)
                    var_1 = getattr(ps, var)[index]
                    ps.ode_fun(0, x_2)
                    var_2 = getattr(ps, var)[index]

                    c_tmp[j] += (var_1 - var_2)/(2*eps)*gain

            c[i, :] = c_tmp
        return c

    def get_mode_idx(self, mode_type=['em', 'non_conj'], damp_threshold=1, sorted=True):
        # Get indices of modes from specified criteria.
        eigs = self.eigs
        idx = np.ones(len(eigs), dtype=bool)
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
    
    import dynpssimpy.dynamic as dps
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
    mode_shape = rev[np.ix_(ps.gen_mdls['GEN'].state_idx['speed'], mode_idx)]

    # Plot mode shape
    fig, ax = plt.subplots(1, mode_shape.shape[1], subplot_kw={'projection': 'polar'})
    for ax_, ms in zip(ax, mode_shape.T):
        dps_plt.plot_mode_shape(ms, ax=ax_, normalize=True)

    plt.show()
