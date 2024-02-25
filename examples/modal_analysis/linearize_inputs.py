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
    
    input_spec = [
        ('gen', 'GEN', 'G1', 'P_m'),
        ('gen', 'GEN', 'G2', 'P_m'),
        ('gen', 'GEN', 'G3', 'P_m'),
        ('gen', 'GEN', 'G4', 'P_m'),
    ]
    B = ps_lin.linearize_inputs_from_spec(input_spec)
    
    plt.imshow(abs(B))
    plt.show()