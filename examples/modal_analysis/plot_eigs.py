import dynpssimpy.dynamic as dps
import dynpssimpy.modal_analysis as dps_mdl
import dynpssimpy.plotting as dps_plt
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    import dynpssimpy.ps_models.n44 as model_data
    model = model_data.load()
    # model['avr'] = {}
    # model['pss'] = {}
    # model['gov'] = {}

    ps = dps.PowerSystemModel(model=model)
    # ps.avr = {}
    # ps.gov = {}
    # ps.pss = {}
    # ps.setup()
    # ps.build_y_bus('lf')
    # ps.power_flow()
    ps.init_dyn_sim()

    # Perform system linearization
    ps_lin = dps_mdl.PowerSystemModelLinearization(ps)
    ps_lin.linearize()
    ps_lin.eigenvalue_decomposition()

    # Alternatively:
    # ps_lin = ps.linearize()

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
