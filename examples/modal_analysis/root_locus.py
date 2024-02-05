import tops.dynamic as dps
import tops.modal_analysis as dps_mdl
import tops.plotting as dps_plt
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    import tops.ps_models.k2a as model_data
    model = model_data.load()
    
    

    # Perform system linearization
    plt.figure()
    for K in np.arange(5, 10, 0.1):
        ps = dps.PowerSystemModel(model=model)
        ps.gen['GEN'].par['H'] = K
        ps.init_dyn_sim()

        ps_lin = dps_mdl.PowerSystemModelLinearization(ps)
        ps_lin.linearize()
        ps_lin.eigenvalue_decomposition()

        plt.scatter(ps_lin.eigs.real, ps_lin.eigs.imag, color=[K/12, K/12, K/12])
    plt.show()

    # Plot eigenvalues
    # dps_plt.plot_eigs(ps_lin.eigs)