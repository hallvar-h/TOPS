import numpy as np
import tops.utility_functions as utils


class PowerSystemModelLinearization:
    def __init__(self, ps):
        self.ps = ps
        self.n = self.ps.n_states
        self.eps = 1e-10
        self.linearize_inputs_v2 = self.linearize_inputs
        self.linearization_ready = False
        self.eigenvalues_ready = False

        self.a = np.empty((self.n,)*2)
        self.b = np.empty((self.n, 0))
        self.c = np.empty((0, self.n))
        self.d = np.empty((0, 0))

        self.lev = np.empty((self.n,)*2, dtype=complex)
        self.rev = np.empty((self.n,)*2, dtype=complex)
        self.eigs = np.empty(self.n, dtype=complex)
        self.freq = np.empty(self.n)
        self.damping = np.empty(self.n)

    def linearize(self, get_eigs=False, ps=None, t0=0, x0=np.array([]), input_description=np.array([]), output_description=np.array([])):
        # Linearizes non-linear ODEs at operating point x0.
        if ps:
            self.ps = ps

        self.x0 = x0 if len(x0) > 0 else self.ps.x0
        self.a = utils.jacobian_num(lambda x: self.ps.ode_fun(t0, x), self.x0, eps=self.eps)
        # self.n = self.a.shape[0]

        if len(input_description) > 0:
            self.linearize_inputs(input_description)
        else:
            self.b = np.zeros((self.n, 0))

        if len(input_description) > 0:
            self.linearize_outputs(output_description)
        else:
            self.c = np.zeros((0, self.n))

        # self.d = np.zeros((self.c.shape[0], self.b.shape[1]))

        if get_eigs:
            self.eigenvalue_decomposition()

        self.linearization_ready = True

    def residues(self, mode_idx):
        if not self.eigenvalues_ready:
            self.eigenvalue_decomposition()
        return self.lev.dot(self.b)[[mode_idx], :]*self.c.dot(self.rev)[:, [mode_idx]]

    def eigenvalue_decomposition(self):
        if not self.linearization_ready:
            self.linearize()

        self.eigs, evs = np.linalg.eig(self.a)

        # Right/left rigenvectors (rev/lev)
        self.rev = evs
        self.lev = np.linalg.inv(self.rev)
        # self.damping = -self.eigs.real / abs(self.eigs)
        self.damping = np.divide(
            -self.eigs.real, abs(self.eigs),
            out=np.zeros_like(self.eigs.real)*np.nan,
            where=self.eigs.real != 0,
        )
        self.freq = self.eigs.imag / (2 * np.pi)

        self.eigenvalues_ready = True

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
        self.b = b
        return b

    def linearize_inputs_v3(self, input_description):
        # Perturbs values in PowerSystemModel-object, as indicated by "input_description", and computes
        # the input matrix (or vector) "b" from the change in states.
        ps = self.ps
        eps = self.eps
        b = np.zeros((len(ps.x0), len(input_description)))
        for i, inp_ in enumerate(input_description):
            inp_(ps, eps)
            f_1 = ps.ode_fun(0, ps.x0)
            inp_(ps, -2*eps)
            f_2 = ps.ode_fun(0, ps.x0)
            inp_(ps, eps)
            b[:, i] = (f_1 - f_2)/(2*eps)
        self.b = b
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
        self.c = c
        return c

    def linearize_outputs_v3(self, output_description):
        # Perturbs states in PowerSystemModel-object, as indicated by "output_description", and computes
        # the output matrix (or vector) "c" from the change in output.
        ps = self.ps
        eps = self.eps
        x = ps.x0.copy()
        c = np.zeros((len(output_description), len(ps.x0)), dtype=complex)
        for i, outp_ in enumerate(output_description):
            c_tmp = np.zeros(ps.n_states, dtype=complex)
            for j in range(ps.n_states):
                x_1 = x.copy()
                x_2 = x.copy()
                x_1[j] += eps
                x_2[j] -= eps

                ps.ode_fun(0, x_1)
                var_1 = outp_(ps)
                ps.ode_fun(0, x_2)
                var_2 = outp_(ps)

                c_tmp[j] += (var_1 - var_2) / (2 * eps)
            c[i, :] = c_tmp
        self.c = c
        return c

    def linearize_outputs_v4(self, output_description):

        ps = self.ps
        eps = self.eps
        t = 0
        x = ps.x0.copy()
        v = ps.v0.copy()

        dtypes = np.zeros(len(output_description), dtype=np.dtype)
        for i, outp_ in enumerate(output_description):
            dtypes[i] = np.dtype(outp_(t, x, v))

        if np.any(dtypes == 'complex128'):
            dtype_c = 'complex128'
        else:
            dtype_c = 'float64'

        c = np.zeros((len(output_description), len(ps.x0)), dtype=dtype_c)
        for i, outp_ in enumerate(output_description):
            c_tmp = np.zeros(ps.n_states, dtype=dtype_c)
            for j in range(ps.n_states):
                x_1 = x.copy()
                x_2 = x.copy()
                x_1[j] += eps
                x_2[j] -= eps

                t_1 = t
                t_2 = t

                ps.ode_fun(0, x_1)
                v_1 = ps.solve_algebraic(0, x_1)
                var_1 = outp_(t_1, x_1, v_1)
                ps.ode_fun(0, x_2)
                v_2 = ps.solve_algebraic(0, x_2)
                var_2 = outp_(t_2, x_2, v_2)

                c_tmp[j] += (var_1 - var_2) / (2 * eps)
            c[i, :] = c_tmp
        self.c = c
        return c

    def get_mode_idx(self, mode_type=['em', 'non_conj'], damp_threshold=1, freq_range=[0.1, 3], sorted=True):
        # Get indices of modes from specified criteria.
        eigs = self.eigs
        idx = np.ones(len(eigs), dtype=bool)
        if not isinstance(mode_type, list):
            mode_type = [mode_type]

        for mt in mode_type:
            if mt == 'em':
                idx *= (abs(eigs.imag) / (2 * np.pi) > freq_range[0]) & (abs(eigs.imag) / (2 * np.pi) < freq_range[1])
            if mt == 'non_conj':
                idx *= eigs.imag >= 0

        idx *= self.damping < damp_threshold

        idx = np.where(idx)[0]
        if sorted:
            idx = idx[np.argsort(self.damping[idx])]
        return idx

    def get_dominant_mode(self):
        em_idx = (0.1 < self.freq) & (self.freq < 2)
        return np.argmin(self.damping)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import tops.plotting as dps_plt
    import tops.dynamic as dps
    import tops.ps_models.k2a as model_data

    import importlib
    importlib.reload(dps)

    ps = dps.PowerSystemModel(model_data.load())
    ps.setup()
    ps.build_y_bus('lf')
    ps.power_flow()
    ps.init_dyn_sim()

    # Perform system linearization
    ps_lin = PowerSystemModelLinearization(ps)
    ps_lin.linearize()
    ps_lin.eigenvalue_decomposition()


    # Plot eigenvalues
    dps_plt.plot_eigs(ps_lin.eigs)

    # Get mode shape for electromechanical modes
    mode_idx = ps_lin.get_mode_idx(['em'], damp_threshold=0.3)
    rev = ps_lin.rev
    mode_shape = rev[np.ix_(ps.gen['GEN'].state_idx_global['speed'], mode_idx)]

    # Plot mode shape
    fig, ax = plt.subplots(1, mode_shape.shape[1], subplot_kw={'projection': 'polar'})
    for ax_, ms in zip(ax, mode_shape.T):
        dps_plt.plot_mode_shape(ms, ax=ax_, normalize=True)

    plt.show()
