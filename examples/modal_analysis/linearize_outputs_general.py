import tops.dynamic as dps
import tops.modal_analysis as dps_mdl
import tops.plotting as dps_plt
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    import tops.ps_models.k2a as model_data
    model = model_data.load()
    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()

    # Perform system linearization
    ps_lin = dps_mdl.PowerSystemModelLinearization(ps)
    ps_lin.linearize()

    A = ps_lin.a
    plt.imshow(abs(A)>0)
    plt.show()
    
    # Make one output function per bus voltage
    output_funs = []
    for i in range(ps.n_bus):
        def output_fun(t, x, v, i=i):
            return v[i]
        output_funs.append(output_fun)

    # Calling the first function should return the voltage at the first bus
    assert output_funs[0](0, ps.x0, ps.v0) == ps.v0[0]
    
    # Find the output matrix
    C = ps_lin.linearize_outputs_v4(output_funs)

    # plt.imshow(abs(C>0))
    plt.imshow(abs(C))
    plt.show()

    # Perform linear simulation
