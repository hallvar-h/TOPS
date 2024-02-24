import tops.dynamic as dps
import tops.modal_analysis as dps_mdl
import tops.plotting as dps_plt
import numpy as np
import matplotlib.pyplot as plt
from tops.utility_functions import lookup_strings

# class PowerSystemModelLinearization(dps_mdl.PowerSystemModelLinearization):


if __name__ == '__main__':

    import tops.ps_models.k2a as model_data
    model = model_data.load()
    del model['gov']
    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()

    ps.gen['GEN'].P_m(ps.x0, ps.v0)
    ps.gen['GEN'].P_m(ps.x0, ps.v0)

    # Perform system linearization
    ps_lin = dps_mdl.PowerSystemModelLinearization(ps)
    ps_lin.linearize()



    A = ps_lin.a
    plt.imshow(abs(A)>0)
    plt.show()
    
    # Make one output function per bus voltage
    input_spec = [
        ('gen', 'GEN', 'G1', 'P_m'),
        ('gen', 'GEN', 'G2', 'P_m'),
        ('gen', 'GEN', 'G3', 'P_m'),
        ('gen', 'GEN', 'G4', 'P_m'),
    ]
    input_funs = []
    for mdl_type, mdl_name, unit_name, input_function in input_spec:
        mdl = getattr(ps, mdl_type)[mdl_name]
        unit_idx = lookup_strings(unit_name, mdl.par['name'])
        def input_fun(ps, eps, mdl=mdl, unit_idx=unit_idx):            
            prev_value = getattr(mdl, input_function)(ps.x0, ps.v0)[unit_idx]
            mdl.set_input(input_function, prev_value + eps, unit_idx)
        input_funs.append(input_fun)

    # ps.gen['GEN'].set_input('P_m', 2, 0)
    # ps.gen['GEN'].set_input(input_function, prev_value + eps, unit_idx)
    
    input_funs[2](ps, 1)
    ps.gen['GEN'].P_m(ps.x0, ps.v0)

    B = ps_lin.linearize_inputs_v3(input_funs)

    # getattr(mdl, output_function)(ps.x0, ps.v0)
        
    # output_funs = []
    # for i in range(ps.n_bus):
    #     def output_fun(t, x, v, i=i):
    #         return v[i]
    #     output_funs.append(output_fun)

    # Calling the first function should return the voltage at the first bus
    # assert output_funs[0](ps.x0, ps.v0) == ps.v0[0]
    # print(input_funs[0](ps.x0, ps.v0))
    
    # Find the output matrix
    # C = ps_lin.linearize_outputs_v5(input_funs)

    # plt.imshow(abs(C>0))
    
    plt.imshow(abs(B))
    plt.show()

    # Perform linear simulation
