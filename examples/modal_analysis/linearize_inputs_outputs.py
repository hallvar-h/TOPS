import tops.dynamic as dps
import tops.modal_analysis as dps_mdl
import tops.plotting as dps_plt
import numpy as np
import matplotlib.pyplot as plt
from tops.utility_functions import lookup_strings


if __name__ == '__main__':

    import tops.ps_models.k2a as model_data
    model = model_data.load()
    del model['gov']
    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()

    # Perform system linearization
    ps_lin = dps_mdl.PowerSystemModelLinearization(ps)
    ps_lin.linearize()
    ps_lin.eigenvalue_decomposition()
    eigs = ps_lin.eigs

    A = ps_lin.a
    
    output_spec = [
        ('gen', 'GEN', 'G1', 'speed'),
        # ('gen', 'GEN', 'G2', 'speed'),
        # ('gen', 'GEN', 'G3', 'speed'),
        # ('gen', 'GEN', 'G4', 'speed'),
    ]

    input_spec = [
        ('gen', 'GEN', 'G1', 'P_m'),
        # ('gen', 'GEN', 'G2', 'P_m'),
        # ('gen', 'GEN', 'G3', 'P_m'),
        # ('gen', 'GEN', 'G4', 'P_m'),
    ]
    B = ps_lin.linearize_inputs_from_spec(input_spec)
    C = ps_lin.linearize_outputs_from_spec(output_spec)
    D = np.zeros((0, 0))

    plt.imshow(abs(A)>0)
    plt.show()
    plt.imshow(abs(B)>0)
    plt.show()
    plt.imshow(abs(C)>0)
    plt.show()

    import scipy
    tf = scipy.signal.ss2tf(A, B, C, D)
    w = np.arange(1, 1000)*0.1
    w, mag, phase = scipy.signal.bode(tf, w)

    from tops.plotting import plot_eigs
    plot_eigs(eigs)
    plt.show()
    
    fig, ax = plt.subplots(2, sharex=True)
    [ax_.grid() for ax_ in ax]
    ax[0].semilogx(w, mag)    # Bode magnitude plot
    # plt.figure()
    ax[1].semilogx(w, phase)  # Bode phase plot
    plt.show()

    t, y = scipy.signal.impulse(tf, T=np.arange(1, 100)*0.1)
    plt.figure()
    plt.plot(t, y)
    plt.show()