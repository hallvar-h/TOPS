import tops.dynamic as dps
import tops.modal_analysis as dps_mdl
import tops.plotting as dps_plt
import numpy as np
import matplotlib.pyplot as plt
from tops.utility_functions import lookup_strings

class PowerSystemModelLinearization(dps_mdl.PowerSystemModelLinearization):
    def linearize_outputs_v5(self, output_description):

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


if __name__ == '__main__':

    import tops.ps_models.k2a as model_data
    model = model_data.load()
    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()

    # Perform system linearization
    ps_lin = PowerSystemModelLinearization(ps)
    ps_lin.linearize()

    A = ps_lin.a
    plt.imshow(abs(A)>0)
    plt.show()
    
    # Make one output function per bus voltage
    output_spec = [
        ('gen', 'GEN', 'G1', 'speed'),
        ('gen', 'GEN', 'G2', 'speed'),
        ('gen', 'GEN', 'G4', 'speed'),
    ]
    output_funs = []
    for mdl_type, mdl_name, unit_name, output_function in output_spec:
        mdl = getattr(ps, mdl_type)[mdl_name]
        unit_idx = lookup_strings(unit_name, mdl.par['name'])
        def output_fun(x, v):
            return getattr(mdl, output_function)[x, v][unit_idx]
        output_funs.append(output_fun)

    # getattr(mdl, output_function)(ps.x0, ps.v0)
        
    # output_funs = []
    # for i in range(ps.n_bus):
    #     def output_fun(t, x, v, i=i):
    #         return v[i]
    #     output_funs.append(output_fun)

    # Calling the first function should return the voltage at the first bus
    # assert output_funs[0](ps.x0, ps.v0) == ps.v0[0]
    print(output_funs[0](ps.x0, ps.v0))
    
    # Find the output matrix
    C = ps_lin.linearize_outputs_v5(output_funs)

    # plt.imshow(abs(C>0))
    plt.imshow(abs(C))
    plt.show()

    # Perform linear simulation
